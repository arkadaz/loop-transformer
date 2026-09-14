"""Plain-text pretraining for the 10M student before Gemma post-training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.model import LoopTransformer, make_compact_10m_config
from src.student_tokenizer import (
    train_compact_tokenizer, tokenizer_from_payload, tokenizer_payload, validate_tokenizer_identity,
)
from src.pretrain_distill import (
    DEFAULT_ANNEAL_CHECKPOINT,
    DEFAULT_PRETRAIN_CHECKPOINT,
    PRETRAIN_QUALITY_GATE_VERSION,
    checkpoint_metadata,
    collate_pairs,
    evaluate_examples,
    load_checkpoint,
    load_tokenizer,
    require_pretrain_ready,
    save_checkpoint,
    seed_everything,
    train,
)


DEFAULT_MIXTURE_CONFIG = "configs/tinystories_foundation.json"
WINDOWING_VERSION = 4
MIXTURE_CONFIG_VERSION = 1
_WIKI40B_MARKER = re.compile(r"_(?:START_ARTICLE|START_SECTION|START_PARAGRAPH|NEWLINE)_")
_FIXED_VALIDATION_SHUFFLE_SEED = 17_291
_DEFAULT_FOUNDATION_EVAL_EXAMPLES = 64
FOUNDATION_ANCHOR_VERSION = 3
GROUNDED_ROLLOUT_VERSION = 1
GROUNDED_ROLLOUT_CASES = 32
GROUNDED_ROLLOUT_TARGET_TOKENS = 16
GROUNDED_ROLLOUT_GENERATED_TOKENS = 16
CONDITIONING_PANEL_VERSION = 1
CONDITIONING_PANEL_CASES = 32
CONDITIONING_TARGET_TOKENS = 8
# A single repeated trigram is normal in a 64-token natural-language
# continuation.  This threshold was calibrated against the immutable gold
# continuation panel: it admits 31/32 gold rows while still rejecting the
# high-repeat loops this check is meant to catch.
HEALTHY_COMPLETION_MAX_REPEATED_TRIGRAM_FRACTION = 0.05
_CONTENT_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_CONTENT_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any",
        "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
        "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
        "with", "would", "you", "your", "yours", "yourself", "yourselves",
    }
)


def text_windows(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    max_input_tokens: int,
    max_target_tokens: int,
    max_examples: int,
    min_document_tokens: int = 1,
    initial_prefix_span: int = 1,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """Turn documents into prefix-to-continuation examples without cross-document leakage.

    A target gets EOS only if it reaches the real end of its document. Intermediate
    windows are deliberately left open so the student does not learn that every
    fixed-size chunk must terminate.
    """
    if max_input_tokens < 1 or max_target_tokens < 2 or min_document_tokens < 1 or initial_prefix_span < 1:
        raise ValueError("max_input_tokens must be positive and max_target_tokens must be at least 2.")
    examples: list[dict[str, Any]] = []
    content_budget = max_target_tokens - 1
    context_caps = (
        max_input_tokens,
        max_input_tokens,
        max(1, max_input_tokens // 2),
        max(1, max_input_tokens // 4),
    )
    iterator = enumerate(texts)
    if progress_label:
        iterator = tqdm(iterator, total=len(texts), desc=progress_label, unit="doc", dynamic_ncols=True)
    for document_index, text in iterator:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < max(2, min_document_tokens):
            continue
        context_cap = context_caps[document_index % len(context_caps)]
        # Vary the first prefix across documents when requested, so short
        # interactive prompts appear during training without crossing docs.
        start = min(1 + document_index % initial_prefix_span, len(token_ids) - 1)
        while start < len(token_ids):
            end = min(len(token_ids), start + content_budget)
            prompt_ids = token_ids[max(0, start - context_cap) : start]
            target_ids = token_ids[start:end]
            examples.append(
                {
                    "prompt": tokenizer.decode(prompt_ids),
                    "target": tokenizer.decode(target_ids),
                    # GPT-2 is byte-level. A window boundary can bisect a
                    # Unicode byte sequence, so the display strings above are
                    # not always safe to tokenize again. Training consumes
                    # these exact IDs through TokenPairs.
                    "prompt_token_ids": list(prompt_ids),
                    "target_token_ids": list(target_ids),
                    "source": "plain_text",
                    "target_eos": end == len(token_ids),
                    "target_token_count": len(target_ids) + int(end == len(token_ids)),
                }
            )
            if len(examples) >= max_examples:
                if progress_label:
                    iterator.close()
                return examples
            start = end
    return examples


def _clean_document(value: Any, *, dataset: str | None = None) -> str:
    """Normalize line endings without flattening code or structured plain text."""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    # Wiki40B keeps corpus-boundary markup in its text field. Those tokens are
    # not natural language and made the student learn literal `_START_*_`
    # continuations during the diagnostic pass, so retain only their intended
    # document/paragraph breaks.
    if dataset == "google/wiki40b":
        text = _WIKI40B_MARKER.sub("\n", text)
    lines = text.split("\n")
    if dataset == "google/wiki40b":
        lines = [line.strip() for line in lines]
    return "\n".join(line.rstrip() for line in lines).strip()


def _row_text(row: Mapping[str, Any], text_field: str) -> Any:
    """Read a simple or dotted text field and fail with a useful error."""
    value: Any = row
    for field in text_field.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(f"Text field {text_field!r} was not found in a dataset row.")
        value = value[field]
    return value


def _row_documents(
    rows: Sequence[dict[str, Any]], label: str, text_field: str, *, dataset: str | None = None
) -> list[str]:
    return [
        text
        for row in tqdm(rows, desc=f"Collecting {label} rows", unit="row", dynamic_ncols=True)
        if (text := _clean_document(_row_text(row, text_field), dataset=dataset))
    ]


def _all_split_documents(args, split: str) -> list[str]:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": split}
    if getattr(args, "dataset_revision", None):
        load_kwargs["revision"] = args.dataset_revision
    rows = load_dataset(args.dataset, args.dataset_config or None, **load_kwargs)
    return _row_documents(rows, split, args.text_field, dataset=args.dataset)


def _stream_partition_is_validation(args, row: Mapping[str, Any], text: str) -> bool:
    """Assign a streamed record to a stable, seed-independent held-out partition."""
    identifier = next(
        (
            str(row[field])
            for field in ("id", "uuid", "url")
            if field in row and row[field] not in (None, "")
        ),
        text,
    )
    revision = args.dataset_revision or "main"
    key = f"{args.dataset}\0{args.dataset_config or ''}\0{revision}\0{identifier}".encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return fraction < args.validation_fraction


def _streaming_documents(
    args,
    split: str,
    *,
    limit: int,
    shuffle_seed: int,
    validation_partition: bool | None,
) -> list[str]:
    """Materialize only a bounded, reproducible portion of an HF streaming split."""
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    rows = load_dataset(args.dataset, args.dataset_config or None, **load_kwargs)
    # A bounded shuffle only rearranges its local buffer. Skip raw training
    # records before shuffling when a continuation needs a later corpus slice.
    train_skip = getattr(args, "streaming_train_skip_documents", 0)
    if split == args.train_split and validation_partition is not True and train_skip:
        rows = rows.skip(train_skip)
    rows = rows.shuffle(seed=shuffle_seed, buffer_size=args.streaming_shuffle_buffer)
    documents: list[str] = []
    skipped_long = skipped_partition = 0
    label = "validation" if validation_partition else "train"
    for row in tqdm(rows, desc=f"Streaming {label} documents", unit="row", dynamic_ncols=True):
        if not isinstance(row, Mapping):
            raise ValueError("A streaming dataset must yield mapping-like rows.")
        token_count = row.get("token_count")
        if token_count is not None:
            try:
                if int(token_count) > args.streaming_max_source_tokens:
                    skipped_long += 1
                    continue
            except (TypeError, ValueError):
                pass
        text = _clean_document(_row_text(row, args.text_field), dataset=args.dataset)
        # Datasets without a token_count column still need a conservative cap
        # before full GPT-2 tokenization in text_windows().
        if not text or len(text) > args.streaming_max_source_tokens * 8:
            skipped_long += 1
            continue
        if validation_partition is not None and _stream_partition_is_validation(args, row, text) != validation_partition:
            skipped_partition += 1
            continue
        documents.append(text)
        if len(documents) >= limit:
            break
    if len(documents) < 2:
        partition = "validation" if validation_partition else "training"
        raise RuntimeError(
            f"Streaming {partition} collection produced fewer than two usable documents "
            f"after scanning the available stream (skipped {skipped_long:,} long and {skipped_partition:,} other-partition rows)."
        )
    return documents


def _streaming_document_splits(args) -> tuple[list[str], list[str]]:
    if args.streaming_shuffle_buffer < 1 or args.streaming_max_source_tokens < 2:
        raise ValueError("streaming-shuffle-buffer must be positive and streaming-max-source-tokens must be at least 2.")
    if getattr(args, "streaming_train_skip_documents", 0) < 0:
        raise ValueError("streaming-train-skip-documents cannot be negative.")
    if args.validation_split:
        return (
            _streaming_documents(
                args,
                args.train_split,
                limit=args.max_documents,
                shuffle_seed=args.seed,
                validation_partition=None,
            ),
            _streaming_documents(
                args,
                args.validation_split,
                limit=args.max_validation_documents,
                shuffle_seed=_FIXED_VALIDATION_SHUFFLE_SEED,
                validation_partition=None,
            ),
        )
    # Two independently shuffled passes make the held-out *selection* fixed
    # while train seed changes only the training sample/order. Hash membership
    # guarantees that no document can appear in both collections.
    return (
        _streaming_documents(
            args,
            args.train_split,
            limit=args.max_documents,
            shuffle_seed=args.seed,
            validation_partition=False,
        ),
        _streaming_documents(
            args,
            args.train_split,
            limit=args.max_validation_documents,
            shuffle_seed=_FIXED_VALIDATION_SHUFFLE_SEED,
            validation_partition=True,
        ),
    )


def _select_documents(documents: Sequence[str], limit: int, shuffle_seed: int) -> list[str]:
    """Take a reproducible training or validation subset after membership is fixed."""
    selected = list(documents)
    random.Random(shuffle_seed).shuffle(selected)
    selected = selected[:limit]
    if len(selected) < 2:
        raise RuntimeError("The selected split did not provide enough non-empty documents.")
    return selected


def load_document_splits(args) -> tuple[list[str], list[str]]:
    if getattr(args, "streaming", False):
        return _streaming_document_splits(args)
    if args.validation_split:
        train = _select_documents(_all_split_documents(args, args.train_split), args.max_documents, args.seed)
        validation = _select_documents(
            _all_split_documents(args, args.validation_split),
            args.max_validation_documents,
            _FIXED_VALIDATION_SHUFFLE_SEED,
        )
        return train, validation
    # Explicit fallback for corpora without a published validation split. Split
    # the full corpus *before* selecting a seed-dependent train subset, so a
    # later continuation cannot move old validation documents into training.
    train_pool, validation_pool = split_documents(
        _all_split_documents(args, args.train_split),
        args.validation_fraction,
        _FIXED_VALIDATION_SHUFFLE_SEED,
    )
    return (
        _select_documents(train_pool, args.max_documents, args.seed),
        _select_documents(validation_pool, args.max_validation_documents, _FIXED_VALIDATION_SHUFFLE_SEED),
    )


def load_mixture_config(path: str | Path) -> list[dict[str, Any]]:
    """Read a small, reproducible list of primary plain-text sources."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Mixture config {config_path} is not valid JSON: {error.msg}.") from error
    if isinstance(payload, Mapping):
        version = payload.get("version", MIXTURE_CONFIG_VERSION)
        sources = payload.get("sources")
    else:
        version = MIXTURE_CONFIG_VERSION
        sources = payload
    if version != MIXTURE_CONFIG_VERSION or not isinstance(sources, list) or not sources:
        raise ValueError("Mixture config needs version 1 and a non-empty sources list.")

    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or not isinstance(source.get("dataset"), str) or not source["dataset"]:
            raise ValueError(f"Mixture source {index} needs a non-empty dataset string.")
        weight = source.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Mixture source {index} needs a finite positive weight.")
        config = source.get("config")
        revision = source.get("revision", "main")
        text_field = source.get("text_field", "text")
        train_split = source.get("train_split", "train")
        validation_split = source.get("validation_split", "validation")
        use_fallback_validation = source.get("use_fallback_validation", False)
        if config is not None and not isinstance(config, str):
            raise ValueError(f"Mixture source {index} config must be a string or null.")
        if revision is not None and not isinstance(revision, str):
            raise ValueError(f"Mixture source {index} revision must be a string or null.")
        if not all(isinstance(value, str) and value for value in (text_field, train_split)):
            raise ValueError(f"Mixture source {index} needs non-empty text_field and train_split strings.")
        if validation_split is not None and not isinstance(validation_split, str):
            raise ValueError(f"Mixture source {index} validation_split must be a string or null.")
        if not isinstance(use_fallback_validation, bool):
            raise ValueError(f"Mixture source {index} use_fallback_validation must be true or false.")
        if not use_fallback_validation and not validation_split:
            raise ValueError(f"Mixture source {index} needs validation_split or use_fallback_validation=true.")
        normalized.append(
            {
                "dataset": source["dataset"],
                "config": config,
                "revision": revision,
                "text_field": text_field,
                "train_split": train_split,
                "validation_split": validation_split,
                "use_fallback_validation": use_fallback_validation,
                "weight": float(weight),
            }
        )
    return normalized


def _weighted_window_budgets(total: int, sources: Sequence[Mapping[str, Any]]) -> list[int]:
    """Allocate exactly `total` windows by source weight with stable tie breaks."""
    if total < 0:
        raise ValueError("Window budget cannot be negative.")
    if not sources:
        return []
    weights = [float(source["weight"]) for source in sources]
    weight_total = sum(weights)
    if not math.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("Mixture weights must sum to a positive finite value.")
    raw = [total * weight / weight_total for weight in weights]
    budgets = [math.floor(value) for value in raw]
    remainder = total - sum(budgets)
    for index in sorted(range(len(sources)), key=lambda item: (-(raw[item] - budgets[item]), item))[:remainder]:
        budgets[index] += 1
    return budgets


def _mixture_source_arguments(args, source: Mapping[str, Any]):
    values = vars(args).copy()
    values.update(
        {
            "dataset": source["dataset"],
            "dataset_config": source["config"],
            "dataset_revision": source["revision"],
            "text_field": source["text_field"],
            "train_split": source["train_split"],
            "validation_split": "" if source["use_fallback_validation"] else source["validation_split"],
        }
    )
    return argparse.Namespace(**values)


def _mixture_source_label(source: Mapping[str, Any]) -> str:
    config = source["config"] or "default"
    return f"{source['dataset']}/{config}"


def _mixture_provenance(args, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "version": MIXTURE_CONFIG_VERSION,
        "config_path": str(args.mixture_config),
        "sources": [dict(source) for source in sources],
    }


def build_mixture_examples(
    args,
    tokenizer: Any,
    sources: Sequence[Mapping[str, Any]],
    *,
    train_budget: int,
    validation_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build and deterministically interleave weighted primary source windows."""
    train_budgets = _weighted_window_budgets(train_budget, sources)
    validation_budgets = _weighted_window_budgets(validation_budget, sources)
    train_groups: list[list[dict[str, Any]]] = []
    validation_groups: list[list[dict[str, Any]]] = []
    realized_sources: list[dict[str, Any]] = []
    for source, source_train_budget, source_validation_budget in zip(sources, train_budgets, validation_budgets):
        source_args = _mixture_source_arguments(args, source)
        train_documents, validation_documents = load_document_splits(source_args)
        label = _mixture_source_label(source)
        train_group = text_windows(
            train_documents,
            tokenizer,
            max_input_tokens=args.max_input_tokens,
            max_target_tokens=args.max_target_tokens,
            max_examples=source_train_budget,
            min_document_tokens=args.min_document_tokens,
            initial_prefix_span=getattr(args, "train_initial_prefix_span", 1),
            progress_label=f"Building {label} train windows",
        ) if source_train_budget else []
        validation_group = text_windows(
            validation_documents,
            tokenizer,
            max_input_tokens=args.max_input_tokens,
            max_target_tokens=args.max_target_tokens,
            max_examples=source_validation_budget,
            min_document_tokens=args.min_document_tokens,
            progress_label=f"Building {label} validation windows",
        ) if source_validation_budget else []
        if len(train_group) != source_train_budget or len(validation_group) != source_validation_budget:
            raise RuntimeError(
                f"Mixture source {label} could not fill its requested window budget "
                f"(train {len(train_group)}/{source_train_budget}, validation {len(validation_group)}/{source_validation_budget})."
            )
        for example in [*train_group, *validation_group]:
            example["source"] = f"plain_text:{label}"
        train_groups.append(train_group)
        validation_groups.append(validation_group)
        realized_sources.append(
            {
                **dict(source),
                "train_window_budget": source_train_budget,
                "validation_window_budget": source_validation_budget,
                "train_windows": len(train_group),
                "validation_windows": len(validation_group),
            }
        )
    return (
        _interleave_example_groups(train_groups, args.seed),
        _interleave_example_groups(validation_groups, _FIXED_VALIDATION_SHUFFLE_SEED),
        {**_mixture_provenance(args, sources), "sources": realized_sources},
    )


def _interleave_example_groups(groups: Sequence[Sequence[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    combined = [example for group in groups for example in group]
    random.Random(seed).shuffle(combined)
    return combined


def split_documents(texts: Sequence[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * fraction))
    return shuffled[validation_size:], shuffled[:validation_size]


def _trim(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _console_text(text: str) -> str:
    """Avoid losing a completed run because a Windows legacy console rejects text."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def _healthy_completion(token_ids: list[int], eos_token_id: int) -> bool:
    # EOS ends a completion.  Do not accidentally score any later padding or
    # sentinel values as generated text.
    tokens: list[int] = []
    for token in token_ids:
        if token == eos_token_id:
            break
        tokens.append(token)
    if len(tokens) < 8:
        return False
    longest_run = run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if previous == current else 1
        longest_run = max(longest_run, run)
    if longest_run >= 4 or len(set(tokens)) / len(tokens) < 0.25:
        return False
    return (
        _repeated_ngram_fraction(tokens, eos_token_id, ngram=3)
        <= HEALTHY_COMPLETION_MAX_REPEATED_TRIGRAM_FRACTION
    )


def _repeated_ngram_fraction(token_ids: Sequence[int], eos_token_id: int, *, ngram: int = 4) -> float:
    """Return the share of generated n-gram opportunities that repeat.

    This is a continuous companion to ``_healthy_completion``: a checkpoint
    can make progress on long-rollout repetition before an entire completion
    becomes healthy.  EOS ends the text and is never treated as a repeated
    content token.
    """
    if ngram < 1:
        raise ValueError("ngram must be positive.")
    content: list[int] = []
    for token in token_ids:
        if token == eos_token_id:
            break
        content.append(token)
    opportunities = len(content) - ngram + 1
    if opportunities <= 0:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    for end in range(ngram - 1, len(content)):
        gram = tuple(content[end + 1 - ngram : end + 1])
        repeats += gram in seen
        seen.add(gram)
    return repeats / opportunities


def _example_token_ids(example: Mapping[str, Any], field: str, tokenizer: Any) -> list[int]:
    """Read exact window IDs when present, with legacy text examples as a fallback."""
    token_field = f"{field}_token_ids"
    ids = example.get(token_field)
    if isinstance(ids, (list, tuple)) and all(isinstance(token, int) for token in ids):
        return list(ids)
    return tokenizer.encode(str(example.get(field, "")), add_special_tokens=False)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode evaluation IDs without making test tokenizers implement every HF option."""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _content_terms(text: str) -> set[str]:
    """Normalize English content terms for deterministic continuation grounding."""
    return {
        term
        for term in _CONTENT_WORD.findall(text.casefold())
        if term not in _CONTENT_STOPWORDS
    }


def _grounded_rollout_spec(
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    stored_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a fixed, suitable rollout panel or reuse its immutable saved identity."""
    if isinstance(stored_spec, Mapping):
        stored_indices = stored_spec.get("case_indices")
        if (
            stored_spec.get("version") == GROUNDED_ROLLOUT_VERSION
            and stored_spec.get("target_tokens") == GROUNDED_ROLLOUT_TARGET_TOKENS
            and stored_spec.get("generated_tokens") == GROUNDED_ROLLOUT_GENERATED_TOKENS
            and isinstance(stored_indices, list)
            and 1 <= len(stored_indices) <= GROUNDED_ROLLOUT_CASES
            and len(set(stored_indices)) == len(stored_indices)
            and all(isinstance(index, int) and 0 <= index < len(examples) for index in stored_indices)
        ):
            # Copy only the stable, serializable schema. This avoids letting a
            # later continuation silently change which anchor cases it scores.
            return {
                "version": GROUNDED_ROLLOUT_VERSION,
                "case_indices": list(stored_indices),
                "target_tokens": GROUNDED_ROLLOUT_TARGET_TOKENS,
                "generated_tokens": GROUNDED_ROLLOUT_GENERATED_TOKENS,
            }

    # Panel identity: prompts with a continuation that introduces new content
    # words, so a rollout has something to say beyond echoing its prompt.
    candidates: list[int] = []
    for index, example in enumerate(examples):
        prompt_ids = _example_token_ids(example, "prompt", tokenizer)
        target_ids = _example_token_ids(example, "target", tokenizer)
        if len(prompt_ids) < 8 or len(target_ids) < GROUNDED_ROLLOUT_TARGET_TOKENS:
            continue
        prompt_terms = _content_terms(_decode_token_ids(tokenizer, prompt_ids))
        target_terms = _content_terms(_decode_token_ids(tokenizer, target_ids[:GROUNDED_ROLLOUT_TARGET_TOKENS]))
        if len(target_terms - prompt_terms) >= 3:
            candidates.append(index)

    count = min(GROUNDED_ROLLOUT_CASES, len(candidates))
    if count == 0:
        indices: list[int] = []
    elif count == 1:
        indices = [candidates[0]]
    else:
        indices = [
            candidates[round(position * (len(candidates) - 1) / (count - 1))]
            for position in range(count)
        ]
    return {
        "version": GROUNDED_ROLLOUT_VERSION,
        "case_indices": indices,
        "target_tokens": GROUNDED_ROLLOUT_TARGET_TOKENS,
        "generated_tokens": GROUNDED_ROLLOUT_GENERATED_TOKENS,
    }


def evaluate_grounded_rollouts(
    model: LoopTransformer,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    device: str,
    *,
    spec: Mapping[str, Any] | None = None,
    generated_tokens: int | None = None,
) -> dict[str, Any]:
    """Greedy-continue a fixed held-out panel and score rollout health and repetition."""
    panel = _grounded_rollout_spec(examples, tokenizer, stored_spec=spec)
    generated_tokens = panel["generated_tokens"] if generated_tokens is None else generated_tokens
    if not isinstance(generated_tokens, int) or not 1 <= generated_tokens < model.config.max_seq_len:
        raise ValueError("generated_tokens must be in [1, model.config.max_seq_len).")
    cases: list[dict[str, Any]] = []
    for index in panel["case_indices"]:
        prompt_ids = _example_token_ids(examples[index], "prompt", tokenizer)[-model.config.max_seq_len :]
        output = model.generate(
            torch.tensor([prompt_ids], device=device),
            thinking_effort="high",
            max_new_tokens=generated_tokens,
            temperature=0.0,
        )[0].tolist()[1:]
        cases.append(
            {
                "index": index,
                "prompt": _trim(_decode_token_ids(tokenizer, prompt_ids)),
                "generated": _trim(_decode_token_ids(tokenizer, output).strip()),
                "healthy": _healthy_completion(output, tokenizer.eos_token_id),
                "repeated_4gram_fraction": _repeated_ngram_fraction(output, tokenizer.eos_token_id),
            }
        )
    count = len(cases)
    return {
        "spec": panel,
        "case_count": count,
        "generated_tokens": generated_tokens,
        "healthy_fraction": sum(case["healthy"] for case in cases) / max(count, 1),
        "mean_repeated_4gram_fraction": sum(case["repeated_4gram_fraction"] for case in cases) / max(count, 1),
        "cases": cases,
    }


def _pretrain_selection_guard(
    model, tokenizer, device, validation_examples, foundation_examples, *,
    foundation_spec, foundation_reference_loss, max_foundation_regression,
    batch_size, max_input_tokens, max_target_tokens,
    rollout_tokens: int = GROUNDED_ROLLOUT_GENERATED_TOKENS,
    selection_strategy: str = "validation",
    selection_validation_tolerance: float = 0.0,
    selection_max_repeat_regression: float = 0.0,
):
    """Keep a candidate only when likelihood and greedy rollouts stay safe.

    ``validation`` preserves the normal lower-CE checkpoint policy.  The
    opt-in ``rollout`` policy is for an explicitly anti-repetition run: it
    ranks candidates by long greedy-rollout health, then repeated-4-gram rate,
    while bounding CE change against the run's starting weights.
    """
    if selection_strategy not in {"validation", "rollout"}:
        raise ValueError("selection_strategy must be 'validation' or 'rollout'.")
    if not isinstance(rollout_tokens, int) or rollout_tokens < 1:
        raise ValueError("rollout_tokens must be a positive integer.")
    if not math.isfinite(selection_validation_tolerance) or selection_validation_tolerance < 0:
        raise ValueError("selection_validation_tolerance must be finite and non-negative.")
    if not math.isfinite(selection_max_repeat_regression) or selection_max_repeat_regression < 0:
        raise ValueError("selection_max_repeat_regression must be finite and non-negative.")
    panels = {
        "current": (validation_examples, _grounded_rollout_spec(validation_examples, tokenizer)),
        "anchor": (foundation_examples, foundation_spec),
    }

    def evaluate_panel(current_model, examples, spec):
        kwargs: dict[str, Any] = {"spec": spec}
        # Keep existing 16-token test doubles and legacy behavior compatible;
        # an override is used only for the explicit long-rollout path.
        if rollout_tokens != GROUNDED_ROLLOUT_GENERATED_TOKENS:
            kwargs["generated_tokens"] = rollout_tokens
        return evaluate_grounded_rollouts(current_model, tokenizer, examples, device, **kwargs)

    baseline_rollouts = {
        name: evaluate_panel(model, examples, spec)
        for name, (examples, spec) in panels.items()
    }
    baseline_health = {name: result["healthy_fraction"] for name, result in baseline_rollouts.items()}
    # Permit four changed cases in a full 32-case panel. This selection guard
    # is separate from, and never weakens, the final 75% readiness threshold.
    max_health_regression = 0.125
    print(
        f"Selection rollout baseline ({rollout_tokens} tokens): current={baseline_health['current']:.0%}, "
        f"anchor={baseline_health['anchor']:.0%}; allow <= {max_health_regression:.1%} drop."
    )

    baseline_validation_loss = baseline_anchor_loss = None
    if selection_strategy == "rollout":
        baseline_validation_loss = evaluate_examples(
            model, validation_examples, tokenizer=tokenizer, device=device,
            batch_size=batch_size, max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens, description="Current selection baseline",
        )
        baseline_anchor_loss = evaluate_examples(
            model, foundation_examples, tokenizer=tokenizer, device=device,
            batch_size=batch_size, max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens, description="Anchor selection baseline",
        )

    def rollout_metric(rollouts: Mapping[str, Mapping[str, float]]) -> float:
        health = [float(rollouts[name]["healthy_fraction"]) for name in ("current", "anchor")]
        repeats = [float(rollouts[name]["mean_repeated_4gram_fraction"]) for name in ("current", "anchor")]
        # Lexicographic in practice: a one-case minimum-health gain (1/32)
        # outweighs every possible secondary-score change.  When health is
        # still zero, the continuous repetition rate can still select genuine
        # progress instead of always rolling back to the start.
        return (1.0 - min(health)) * 128.0 + (1.0 - sum(health) / len(health)) * 2.0 + sum(repeats) / len(repeats)

    def select(current_model, validation_loss):
        anchor_loss = evaluate_examples(
            current_model, foundation_examples, tokenizer=tokenizer, device=device,
            batch_size=batch_size, max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens, description="Anchor selection evaluation",
        )
        reference_regression = anchor_loss - foundation_reference_loss
        allowed = reference_regression <= max_foundation_regression
        details = {
            "anchor_loss": anchor_loss,
            "anchor_reference_regression": reference_regression,
            "max_health_regression": max_health_regression,
        }
        candidate_rollouts: dict[str, Mapping[str, float]] = {}
        for name, (examples, spec) in panels.items():
            rollout = evaluate_panel(current_model, examples, spec)
            candidate_rollouts[name] = rollout
            health = rollout["healthy_fraction"]
            regression = baseline_health[name] - health
            allowed = allowed and regression <= max_health_regression
            details.update({
                f"{name}_healthy_fraction": health,
                f"{name}_baseline_healthy_fraction": baseline_health[name],
                f"{name}_health_regression": regression,
            })
            if selection_strategy == "rollout":
                repeat_fraction = rollout["mean_repeated_4gram_fraction"]
                repeat_baseline = baseline_rollouts[name]["mean_repeated_4gram_fraction"]
                repeat_regression = repeat_fraction - repeat_baseline
                allowed = allowed and repeat_regression <= selection_max_repeat_regression
                details.update({
                    f"{name}_mean_repeated_4gram_fraction": repeat_fraction,
                    f"{name}_baseline_mean_repeated_4gram_fraction": repeat_baseline,
                    f"{name}_repeat4_regression": repeat_regression,
                })
        if selection_strategy == "rollout":
            assert baseline_validation_loss is not None and baseline_anchor_loss is not None
            validation_regression = validation_loss - baseline_validation_loss
            anchor_run_regression = anchor_loss - baseline_anchor_loss
            allowed = (
                allowed
                and validation_regression <= selection_validation_tolerance
                and anchor_run_regression <= selection_validation_tolerance
            )
            details.update({
                "current_baseline_loss": baseline_validation_loss,
                "current_loss_regression": validation_regression,
                "anchor_baseline_loss": baseline_anchor_loss,
                "anchor_loss_regression": anchor_run_regression,
                "selection_validation_tolerance": selection_validation_tolerance,
                "selection_max_repeat_regression": selection_max_repeat_regression,
            })
            return allowed, rollout_metric(candidate_rollouts), details
        return allowed, validation_loss, details

    if selection_strategy == "rollout":
        # ``train`` needs the starting score in the same units as candidate
        # scores; otherwise it would accept the first safe epoch by accident.
        select.initial_selection_metric = rollout_metric(baseline_rollouts)

    return select


def _conditioning_panel_spec(examples, tokenizer, *, stored_spec=None):
    """Freeze exact prompts/targets and seven distinct, low-overlap distractors."""
    if stored_spec is not None:
        if (
            stored_spec.get("version") != CONDITIONING_PANEL_VERSION
            or stored_spec.get("target_tokens") != CONDITIONING_TARGET_TOKENS
            or not isinstance(stored_spec.get("cases"), list)
            or len(stored_spec["cases"]) > CONDITIONING_PANEL_CASES
        ):
            raise ValueError("Invalid frozen prompt-conditioning panel.")
        for case in stored_spec["cases"]:
            sequences = [case.get("prompt_token_ids"), case.get("target_token_ids"), *case.get("distractors", [])]
            if (
                len(case.get("distractors", [])) != 7
                or len(case.get("target_token_ids", [])) != CONDITIONING_TARGET_TOKENS
                or any(not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids) for ids in sequences)
                or len({tuple(ids) for ids in [sequences[0], *sequences[2:]]}) != 8
            ):
                raise ValueError("Invalid frozen prompt-conditioning case.")
        return stored_spec
    candidates, seen = [], set()
    for example in examples:
        prompt = _example_token_ids(example, "prompt", tokenizer)
        target = _example_token_ids(example, "target", tokenizer)[:CONDITIONING_TARGET_TOKENS]
        if len(prompt) >= 8 and len(target) == CONDITIONING_TARGET_TOKENS and tuple(prompt) not in seen:
            seen.add(tuple(prompt))
            candidates.append({"prompt_token_ids": prompt, "target_token_ids": target})
    count = min(CONDITIONING_PANEL_CASES, len(candidates))
    selected = [candidates[round(i * (len(candidates) - 1) / max(count - 1, 1))] for i in range(count)]
    terms = [_content_terms(_decode_token_ids(tokenizer, case["prompt_token_ids"])) for case in selected]
    cases = []
    if count >= 8:
        for i, case in enumerate(selected):
            # Lexical overlap first, length mismatch second, stable cyclic ties.
            unrelated = sorted(
                (j for j in range(count) if j != i),
                key=lambda j: (
                    len(terms[i] & terms[j]) / max(len(terms[i] | terms[j]), 1),
                    abs(math.log(len(case["prompt_token_ids"]) / len(selected[j]["prompt_token_ids"]))),
                    (j - i) % count,
                ),
            )[:7]
            cases.append({**case, "distractors": [selected[j]["prompt_token_ids"] for j in unrelated]})
    return {"version": CONDITIONING_PANEL_VERSION, "target_tokens": CONDITIONING_TARGET_TOKENS, "cases": cases}


@torch.inference_mode()
def evaluate_prompt_conditioning(model, tokenizer, examples, device, *, spec=None):
    """Rank the true prompt by NLL of the same eight teacher-forced target IDs."""
    panel = _conditioning_panel_spec(examples, tokenizer, stored_spec=spec)
    cases = []
    was_training = model.training
    model.eval()
    try:
        for case in panel["cases"]:
            prompts = [case["prompt_token_ids"], *case["distractors"]]
            if any(len(prompt) > model.config.max_seq_len for prompt in prompts):
                raise ValueError("Frozen conditioning prompt exceeds model context; do not silently truncate its identity.")
            batch = collate_pairs(
                [(prompt, case["target_token_ids"]) for prompt in prompts],
                tokenizer.pad_token_id, model.config.decoder_start_token_id,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                batch["input_ids"], batch["decoder_ids"],
                input_attention_mask=batch["input_mask"], target_attention_mask=batch["target_mask"],
                thinking_effort="high",
            )
            nll = F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(), reduction="none")
            losses = nll.reshape(8, CONDITIONING_TARGET_TOKENS).mean(dim=1).tolist()
            if not all(math.isfinite(loss) for loss in losses):
                raise ValueError("Non-finite prompt-conditioning loss.")
            cases.append({
                "correct_nll": losses[0], "distractor_nlls": losses[1:],
                "rank_win": losses[0] < min(losses[1:]),
                "margin": sum(losses[1:]) / 7 - losses[0],
            })
    finally:
        model.train(was_training)
    return {
        "spec": panel, "case_count": len(cases), "cases": cases,
        "win_fraction": sum(case["rank_win"] for case in cases) / max(len(cases), 1),
        "mean_margin": sum(case["margin"] for case in cases) / max(len(cases), 1),
    }


def _conditioning_ready(result, threshold):
    return (
        result["case_count"] == CONDITIONING_PANEL_CASES
        and result["win_fraction"] >= threshold
        and math.isfinite(result["mean_margin"])
        and result["mean_margin"] > 0
    )


def _fixed_evaluation_examples(
    examples: Sequence[dict[str, Any]], count: int, provenance: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Store a small, deterministic foundation set to detect annealing regression."""
    if count < 1:
        raise ValueError("foundation-eval-examples must be positive.")
    if not examples:
        raise ValueError("Cannot create a fixed evaluation set from no examples.")
    count = min(count, len(examples))
    indices = sorted({round(index * (len(examples) - 1) / max(count - 1, 1)) for index in range(count)})
    return [
        {
            "prompt": examples[index]["prompt"],
            "target": examples[index]["target"],
            "target_eos": bool(examples[index].get("target_eos", True)),
            "source": "foundation_heldout",
            "foundation_provenance": dict(provenance),
            **{
                field: list(examples[index][field])
                for field in ("prompt_token_ids", "target_token_ids")
                if isinstance(examples[index].get(field), (list, tuple))
            },
        }
        for index in indices
    ]


def _recipe(args) -> dict[str, Any]:
    return {
        "windowing_version": WINDOWING_VERSION,
        "mixture": _mixture_provenance(args, args.mixture_sources),
        "streaming": args.streaming,
        "streaming_shuffle_buffer": args.streaming_shuffle_buffer,
        "streaming_max_source_tokens": args.streaming_max_source_tokens,
        "streaming_train_skip_documents": args.streaming_train_skip_documents,
        "train_initial_prefix_span": args.train_initial_prefix_span,
        "max_documents": args.max_documents,
        "max_validation_documents": args.max_validation_documents,
        "max_examples": args.max_examples,
        "max_input_tokens": args.max_input_tokens,
        "max_target_tokens": args.max_target_tokens,
        "prefix_loss_weight": args.prefix_loss_weight,
        "prefix_loss_tokens": args.prefix_loss_tokens,
        "decoder_prompt_prefill_tokens": getattr(args, "decoder_prompt_prefill_tokens", None),
        "rollout_tokens": args.rollout_tokens,
        "selection_strategy": args.selection_strategy,
        "selection_rollout_tokens": args.selection_rollout_tokens,
        "selection_validation_tolerance": args.selection_validation_tolerance,
        "selection_max_repeat_regression": args.selection_max_repeat_regression,
        "min_document_tokens": args.min_document_tokens,
        "validation_fraction": args.validation_fraction,
        "foundation_eval_examples": args.foundation_eval_examples,
        "seed": args.seed,
    }


def _partial_checkpoint_path(output: str) -> str:
    path = Path(output)
    return str(path.with_name(f"{path.stem}.partial{path.suffix}"))


def _stored_foundation_anchor(checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the immutable anchor from a checkpoint, if it has one."""
    stored = checkpoint.get("foundation_anchor")
    if isinstance(stored, Mapping) and isinstance(stored.get("examples"), list) and stored["examples"]:
        return dict(stored)
    return None


def _foundation_anchor(
    parent_anchor: Mapping[str, Any] | None,
    validation_examples: Sequence[dict[str, Any]],
    args,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep a parent held-out anchor when a foundation run changes corpus."""
    if parent_anchor is not None:
        return parent_anchor["examples"], dict(parent_anchor.get("provenance") or {"dataset": "parent checkpoint"})
    provenance = {"dataset": "primary_mixture", "mixture": _mixture_provenance(args, args.mixture_sources)}
    return _fixed_evaluation_examples(validation_examples, args.foundation_eval_examples, provenance), provenance


def _foundation_reference(parent_anchor: Mapping[str, Any] | None, baseline_loss: float) -> dict[str, Any]:
    """Separate a pending fresh reference from a frozen trained reference."""
    if not math.isfinite(baseline_loss):
        raise ValueError("Foundation baseline loss must be finite.")
    if parent_anchor is None:
        return {"reference_state": "pending", "reference_loss": None}
    state = parent_anchor.get("reference_state")
    if state not in {"pending", "frozen"}:
        raise ValueError("Invalid foundation reference state.")
    fields = {
        key: parent_anchor[key]
        for key in ("reference_state", "reference_loss", "reference_source", "reference_checkpoint")
        if key in parent_anchor
    }
    if state == "pending":
        if fields.get("reference_loss") is not None:
            raise ValueError("A pending foundation reference cannot have a frozen loss.")
        fields["reference_loss"] = None
        return fields
    reference_loss = fields.get("reference_loss")
    if not isinstance(reference_loss, (int, float)) or not math.isfinite(reference_loss):
        raise ValueError("A frozen foundation reference needs a finite loss.")
    return fields


def _freeze_foundation_reference(
    anchor: dict[str, Any], selected_loss: float, selected_optimizer_steps: int, checkpoint: str
) -> None:
    """Freeze only selected trained weights; partial/random weights stay pending."""
    if anchor["reference_state"] == "pending" and selected_optimizer_steps > 0:
        if not math.isfinite(selected_loss):
            raise ValueError("Selected foundation loss must be finite.")
        anchor.update(
            reference_state="frozen",
            reference_loss=selected_loss,
            reference_source="selected_checkpoint",
            reference_checkpoint=checkpoint,
        )


def _validate_resumed_architecture(model: LoopTransformer, requested: str | None) -> None:
    """Never reinterpret resumed weights as a different architecture."""
    config = model.config
    compact = config.vocab_size == 8192 and config.d_model == 256 and config.d_ff == 1216 and config.num_decoder_layers == 6
    if requested is not None and not compact:
        raise ValueError(
            f"--architecture {requested} conflicts with the checkpoint's architecture. "
            "Omit --init-checkpoint to train a fresh architecture."
        )


def _fresh_tokenizer(args, mixture_sources):
    requested = getattr(args, "tokenizer", None) or "compact-bpe"
    vocab_size = getattr(args, "tokenizer_vocab_size", None)
    if requested != "compact-bpe":
        tokenizer = load_tokenizer() if requested == "gpt2" else tokenizer_from_payload(
            json.loads(Path(requested).read_text(encoding="utf-8"))
        )
        if vocab_size is not None and len(tokenizer) != vocab_size:
            raise ValueError("--tokenizer-vocab-size differs from the selected tokenizer.")
        return tokenizer
    limit = getattr(args, "tokenizer_max_documents", 10_000)
    if limit < 2:
        raise ValueError("--tokenizer-max-documents must be at least 2.")
    print(f"Training byte BPE on up to {limit:,} training documents (held-out documents excluded).")

    def training_texts():
        for source, budget in zip(mixture_sources, _weighted_window_budgets(limit, mixture_sources)):
            if budget < 2:
                continue
            sample_args = _mixture_source_arguments(args, source)
            sample_args.max_documents = min(args.max_documents, budget)
            sample_args.max_validation_documents = 2
            documents, _ = load_document_splits(sample_args)
            yield from documents

    tokenizer = train_compact_tokenizer(training_texts(), vocab_size or 8192)
    if len(tokenizer) != (vocab_size or 8192):
        raise ValueError("Tokenizer training sample is too small to fill the requested vocabulary. Increase --tokenizer-max-documents.")
    print(f"Trained {len(tokenizer):,}-token byte BPE; it will be embedded in every checkpoint.")
    return tokenizer


def _validate_resumed_tokenizer(args, tokenizer) -> None:
    requested = getattr(args, "tokenizer", None)
    if requested == "compact-bpe":
        raise ValueError("--tokenizer compact-bpe trains a new tokenizer. Omit it when resuming an embedded tokenizer.")
    if requested is not None:
        explicit = load_tokenizer() if requested == "gpt2" else tokenizer_from_payload(
            json.loads(Path(requested).read_text(encoding="utf-8"))
        )
        validate_tokenizer_identity(tokenizer, explicit)
    vocab_size = getattr(args, "tokenizer_vocab_size", None)
    if vocab_size is not None and vocab_size != len(tokenizer):
        raise ValueError("--tokenizer-vocab-size differs from the checkpoint tokenizer.")


def run(args) -> None:
    stage = getattr(args, "stage", "pretrain")
    if stage not in {"pretrain", "anneal"}:
        raise ValueError(f"Unsupported plain-text stage {stage!r}.")
    if args.max_foundation_regression < 0 or args.foundation_eval_examples < 1:
        raise ValueError("max-foundation-regression cannot be negative and foundation-eval-examples must be positive.")
    if args.train_initial_prefix_span < 1:
        raise ValueError("train-initial-prefix-span must be positive.")
    if args.streaming_train_skip_documents < 0:
        raise ValueError("streaming-train-skip-documents cannot be negative.")
    if not 0 <= args.min_healthy_probe_fraction <= 1:
        raise ValueError("minimum healthy probe fraction must be between 0 and 1.")
    if not 0 <= args.min_conditioning_win_fraction <= 1:
        raise ValueError("minimum conditioning win fraction must be between 0 and 1.")
    if args.rollout_tokens < 1:
        raise ValueError("rollout-tokens must be positive.")
    if args.selection_rollout_tokens is not None and args.selection_rollout_tokens < 1:
        raise ValueError("selection-rollout-tokens must be positive when provided.")
    if args.selection_validation_tolerance < 0:
        raise ValueError("selection-validation-tolerance cannot be negative.")
    if args.selection_max_repeat_regression < 0:
        raise ValueError("selection-max-repeat-regression cannot be negative.")
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Keep the parsed source specs on args so checkpoint provenance is fully
    # self-contained instead of relying on a local JSON file surviving.
    mixture_sources = args.mixture_sources = load_mixture_config(args.mixture_config)
    parent_anchor: dict[str, Any] | None = None
    resume_training_state: dict[str, Any] | None = None
    if args.init_checkpoint:
        expected_stages = ("pretrain", "anneal") if stage == "anneal" else "pretrain"
        model, tokenizer = load_checkpoint(args.init_checkpoint, device, expected_stage=expected_stages)
        _validate_resumed_architecture(model, getattr(args, "architecture", None))
        _validate_resumed_tokenizer(args, tokenizer)
        checkpoint_info = checkpoint_metadata(args.init_checkpoint)
        parent_anchor = _stored_foundation_anchor(checkpoint_info)
        # Foundation-to-annealing is a deliberate optimization boundary. An
        # annealing continuation, however, can retain the AdamW state from its
        # own previous annealing checkpoint.
        if checkpoint_info.get("stage") == stage:
            candidate_state = checkpoint_info.get("training_state")
            if candidate_state is not None:
                if not isinstance(candidate_state, dict):
                    raise ValueError("Checkpoint training state is malformed.")
                resume_training_state = candidate_state
        candidate_quality = checkpoint_info.get("pretrain_quality")
        if stage == "anneal":
            # Check eligibility before loading a potentially enormous streamed
            # corpus, then retain the foundation sample for a regression check.
            require_pretrain_ready(args.init_checkpoint)
            if parent_anchor is None:
                raise ValueError(
                    "Annealing needs a gated foundation checkpoint with the current fixed foundation evaluation set. "
                    "Run one completed foundation continuation first."
                )
        continuation = " with AdamW state" if resume_training_state is not None else " (weights only)"
        print(f"Resuming {stage} from {args.init_checkpoint}{continuation}.")
    elif stage == "anneal":
        raise ValueError("Annealing requires a pretrain checkpoint that passed the foundation quality gate.")
    else:
        tokenizer = _fresh_tokenizer(args, mixture_sources)
        model = LoopTransformer(
            make_compact_10m_config(
                len(tokenizer),
                decoder_start_token_id=(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id),
                eos_token_id=tokenizer.eos_token_id,
            )
        ).to(device)
        if getattr(args, "tokenizer", None) != "gpt2":
            model._loop_tokenizer_payload = tokenizer_payload(tokenizer)
    requested_prefill = getattr(args, "decoder_prompt_prefill_tokens", None)
    if requested_prefill is not None:
        model.config.decoder_prompt_prefill_tokens = requested_prefill
    prefill = model.config.decoder_prompt_prefill_tokens
    if prefill < 0 or prefill >= model.config.max_seq_len:
        raise ValueError("decoder-prompt-prefill-tokens must be in [0, max_seq_len).")
    rollout_tokens = args.rollout_tokens
    selection_rollout_tokens = args.selection_rollout_tokens or rollout_tokens
    if max(rollout_tokens, selection_rollout_tokens) > model.config.max_seq_len - max(prefill, 1):
        raise ValueError("rollout token counts plus decoder prompt prefill must fit the model context length.")
    print(f"Decoder context: {f'{prefill} prompt tokens' if prefill else 'BOS'}.")
    labels = ", ".join(_mixture_source_label(source) for source in mixture_sources)
    print(f"Loading primary mixture ({labels}) for plain-text pretraining on {device}.")
    validation_budget = args.max_validation_examples or max(1, round(args.max_examples * args.validation_fraction))
    train_examples, validation_examples, mixture_provenance = build_mixture_examples(
        args,
        tokenizer,
        mixture_sources,
        train_budget=args.max_examples,
        validation_budget=validation_budget,
    )
    if not train_examples or not validation_examples:
        raise RuntimeError("Increase max_documents or choose a text source with longer documents.")
    supervised_tokens = sum(int(example["target_token_count"]) for example in train_examples)
    natural_endings = sum(bool(example["target_eos"]) for example in train_examples)
    checkpoint_sources = [source["dataset"] for source in mixture_sources]
    print(
        f"Pretraining on {len(train_examples):,} windows / {supervised_tokens:,} supervised tokens "
        f"({natural_endings:,} natural EOS labels) validating on {len(validation_examples):,}. "
        f"Parameters: {model.parameter_count:,}"
    )
    foundation_evaluation_examples, foundation_evaluation_provenance = _foundation_anchor(
        parent_anchor,
        validation_examples,
        args,
    )
    stored_token_limits = parent_anchor.get("token_limits") if parent_anchor is not None else None
    foundation_token_limits = {
        "max_input_tokens": int(stored_token_limits.get("max_input_tokens", args.max_input_tokens))
        if isinstance(stored_token_limits, Mapping)
        else args.max_input_tokens,
        "max_target_tokens": int(stored_token_limits.get("max_target_tokens", args.max_target_tokens))
        if isinstance(stored_token_limits, Mapping)
        else args.max_target_tokens,
    }
    # Calculate the anchor from the exact parent weights rather than trusting a
    # stale number in metadata. This makes a corpus switch auditable.
    foundation_baseline_loss = evaluate_examples(
        model,
        foundation_evaluation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=foundation_token_limits["max_input_tokens"],
        max_target_tokens=foundation_token_limits["max_target_tokens"],
        description="Foundation anchor baseline",
    )
    foundation_baseline_first_token_loss = evaluate_examples(
        model,
        foundation_evaluation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=foundation_token_limits["max_input_tokens"],
        max_target_tokens=foundation_token_limits["max_target_tokens"],
        description="Foundation anchor first-token baseline",
        first_token_only=True,
    )
    print(f"foundation anchor baseline={foundation_baseline_loss:.4f}")
    reference = _foundation_reference(parent_anchor, foundation_baseline_loss)
    # A fresh/partial run may use its starting weights for local selection,
    # but that provisional loss must never become its descendant's anchor.
    foundation_reference_loss = (
        reference["reference_loss"] if reference["reference_state"] == "frozen" else foundation_baseline_loss
    )
    stored_grounded_rollout = parent_anchor.get("grounded_rollout") if parent_anchor is not None else None
    foundation_grounded_rollout_spec = _grounded_rollout_spec(
        foundation_evaluation_examples,
        tokenizer,
        stored_spec=stored_grounded_rollout if isinstance(stored_grounded_rollout, Mapping) else None,
    )
    foundation_conditioning_spec = _conditioning_panel_spec(
        foundation_evaluation_examples, tokenizer,
        stored_spec=parent_anchor.get("conditioning_panel") if parent_anchor is not None else None,
    )
    foundation_anchor = {
        "version": FOUNDATION_ANCHOR_VERSION,
        "examples": foundation_evaluation_examples,
        "provenance": foundation_evaluation_provenance,
        **reference,
        "token_limits": foundation_token_limits,
        # Persist the panel identity with the immutable examples. Old
        # checkpoints get one at their next completed continuation.
        "grounded_rollout": foundation_grounded_rollout_spec,
        "conditioning_panel": foundation_conditioning_spec,
    }
    partial_output = _partial_checkpoint_path(args.output)

    def checkpoint_progress(
        current_model: LoopTransformer,
        history: list[dict[str, float]],
        training_state: dict[str, Any],
    ) -> None:
        save_checkpoint(
            partial_output,
            current_model,
            checkpoint_sources,
            history,
            stage=stage,
            parent_checkpoint=args.init_checkpoint,
            # A partial run has not yet been evaluated. Never let an older
            # readiness result authorize changed weights for post-training.
            pretrain_quality=None,
            foundation_anchor=foundation_anchor,
            training_state=training_state,
        )

    select_without_forgetting = _pretrain_selection_guard(
        model, tokenizer, device, validation_examples, foundation_evaluation_examples,
        foundation_spec=foundation_grounded_rollout_spec,
        foundation_reference_loss=foundation_reference_loss,
        max_foundation_regression=args.max_foundation_regression,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=foundation_token_limits["max_input_tokens"],
        max_target_tokens=foundation_token_limits["max_target_tokens"],
        rollout_tokens=selection_rollout_tokens,
        selection_strategy=args.selection_strategy,
        selection_validation_tolerance=args.selection_validation_tolerance,
        selection_max_repeat_regression=args.selection_max_repeat_regression,
    )

    model, history = train(
        model,
        train_examples,
        validation_examples,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
        seed=args.seed,
        checkpoint_every_steps=args.save_every_steps,
        on_checkpoint=checkpoint_progress,
        thinking_efforts=(args.thinking_effort,),
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        resume_training_state=resume_training_state,
        selection_callback=select_without_forgetting,
        initial_selection_metric=getattr(select_without_forgetting, "initial_selection_metric", None),
        prefix_loss_weight=args.prefix_loss_weight,
        prefix_loss_tokens=args.prefix_loss_tokens,
    )
    run_baseline = history[0]["baseline_validation_loss"]
    best_validation = evaluate_examples(
        model,
        validation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
        description="Selected checkpoint validation",
    )
    selected_first_token_loss = evaluate_examples(
        model,
        validation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
        description="Selected checkpoint first-token validation",
        first_token_only=True,
    )
    current_rollouts = evaluate_grounded_rollouts(
        model,
        tokenizer,
        validation_examples,
        device,
        generated_tokens=rollout_tokens,
    )
    foundation_rollouts = evaluate_grounded_rollouts(
        model,
        tokenizer,
        foundation_evaluation_examples,
        device,
        spec=foundation_grounded_rollout_spec,
        generated_tokens=rollout_tokens,
    )
    current_conditioning = evaluate_prompt_conditioning(model, tokenizer, validation_examples, device)
    foundation_conditioning = evaluate_prompt_conditioning(
        model, tokenizer, foundation_evaluation_examples, device, spec=foundation_conditioning_spec,
    )
    probes = current_rollouts["cases"]
    foundation_probes = foundation_rollouts["cases"]
    healthy_fraction = current_rollouts["healthy_fraction"]
    foundation_healthy_fraction = foundation_rollouts["healthy_fraction"]
    foundation_evaluation_loss = evaluate_examples(
        model,
        foundation_evaluation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=foundation_token_limits["max_input_tokens"],
        max_target_tokens=foundation_token_limits["max_target_tokens"],
        description="Foundation regression evaluation",
    )
    foundation_first_token_loss = evaluate_examples(
        model,
        foundation_evaluation_examples,
        tokenizer=tokenizer,
        device=device,
        batch_size=min(args.batch_size, 64),
        max_input_tokens=foundation_token_limits["max_input_tokens"],
        max_target_tokens=foundation_token_limits["max_target_tokens"],
        description="Foundation regression first-token evaluation",
        first_token_only=True,
    )
    selected_training_state = getattr(model, "_loop_training_state", None) or {}
    _freeze_foundation_reference(
        foundation_anchor,
        foundation_evaluation_loss,
        int(selected_training_state.get("completed_optimizer_steps", 0)),
        args.output,
    )
    if foundation_anchor["reference_state"] == "frozen":
        foundation_reference_loss = foundation_anchor["reference_loss"]
    foundation_regression = foundation_evaluation_loss - foundation_baseline_loss
    foundation_reference_regression = foundation_evaluation_loss - foundation_reference_loss
    foundation_regression_ok = (
        foundation_anchor["reference_state"] == "frozen"
        and foundation_reference_regression <= args.max_foundation_regression
    )
    loss_ready = best_validation <= args.max_validation_loss
    foundation_loss_ready = foundation_evaluation_loss <= args.max_validation_loss
    healthy_probes_ready = healthy_fraction >= args.min_healthy_probe_fraction
    foundation_healthy_probes_ready = foundation_healthy_fraction >= args.min_healthy_probe_fraction
    conditioning_ready = _conditioning_ready(current_conditioning, args.min_conditioning_win_fraction)
    foundation_conditioning_ready = _conditioning_ready(foundation_conditioning, args.min_conditioning_win_fraction)
    quality = {
        "gate_version": PRETRAIN_QUALITY_GATE_VERSION,
        "passed": (
            loss_ready
            and foundation_loss_ready
            and healthy_probes_ready
            and foundation_healthy_probes_ready
            and conditioning_ready
            and foundation_conditioning_ready
            and foundation_regression_ok
        ),
        "initial_validation_loss": run_baseline,
        "run_baseline_validation_loss": run_baseline,
        "best_validation_loss": best_validation,
        "loss_ready": loss_ready,
        "probe_count": len(probes),
        "healthy_probe_fraction": healthy_fraction,
        "rollout_tokens": rollout_tokens,
        "mean_repeated_4gram_fraction": current_rollouts["mean_repeated_4gram_fraction"],
        "continuation_probes": probes,
        "healthy_probes_ready": healthy_probes_ready,
        "grounded_rollout_spec": current_rollouts["spec"],
        "conditioning": current_conditioning,
        "conditioning_ready": conditioning_ready,
        "conditioning_min_win_fraction": args.min_conditioning_win_fraction,
        "foundation_probe_count": len(foundation_probes),
        "foundation_healthy_probe_fraction": foundation_healthy_fraction,
        "foundation_mean_repeated_4gram_fraction": foundation_rollouts["mean_repeated_4gram_fraction"],
        "foundation_continuation_probes": foundation_probes,
        "foundation_healthy_probes_ready": foundation_healthy_probes_ready,
        "foundation_grounded_rollout_spec": foundation_rollouts["spec"],
        "foundation_conditioning": foundation_conditioning,
        "foundation_conditioning_ready": foundation_conditioning_ready,
        "foundation_evaluation_examples": foundation_evaluation_examples,
        "foundation_evaluation_provenance": foundation_evaluation_provenance,
        "foundation_evaluation_baseline_loss": foundation_baseline_loss,
        "foundation_evaluation_loss": foundation_evaluation_loss,
        "foundation_first_token_baseline_loss": foundation_baseline_first_token_loss,
        "foundation_first_token_loss": foundation_first_token_loss,
        "selected_first_token_loss": selected_first_token_loss,
        "foundation_regression": foundation_regression,
        "foundation_reference_loss": foundation_reference_loss,
        "foundation_reference_regression": foundation_reference_regression,
        "foundation_regression_ok": foundation_regression_ok,
        "foundation_loss_ready": foundation_loss_ready,
        "foundation_evaluation_token_limits": foundation_token_limits,
        "mixture_provenance": mixture_provenance,
        "recipe": _recipe(args),
    }
    save_checkpoint(
        args.output,
        model,
        checkpoint_sources,
        history,
        stage=stage,
        parent_checkpoint=args.init_checkpoint,
        pretrain_quality=quality,
        foundation_anchor=foundation_anchor,
        training_state=getattr(model, "_loop_training_state", None),
    )
    print("Held-out rollout samples:")
    for number, probe in enumerate(probes[:8], 1):
        status = "healthy" if probe["healthy"] else "looping"
        print(_console_text(f"  {number}. [{status}] {_trim(probe['prompt'], 64)!r} -> {probe['generated']!r}"))
    print(
        f"Rollout health ({rollout_tokens} greedy tokens): current={healthy_fraction:.0%}, "
        f"repeat4={current_rollouts['mean_repeated_4gram_fraction']:.0%}; "
        f"anchor={foundation_healthy_fraction:.0%}, "
        f"repeat4={foundation_rollouts['mean_repeated_4gram_fraction']:.0%}."
    )
    print(
        f"First-token CE: current={selected_first_token_loss:.3f}; "
        f"anchor={foundation_first_token_loss:.3f} (was {foundation_baseline_first_token_loss:.3f})."
    )
    for label, result in (("current", current_conditioning), ("anchor", foundation_conditioning)):
        print(
            f"Prompt conditioning ({label}): {result['win_fraction']:.1%} strict rank wins "
            f"over {result['case_count']}/{CONDITIONING_PANEL_CASES} cases; "
            f"mean NLL margin={result['mean_margin']:+.4f} (positive favors the correct prompt)."
        )
    state = "PASSED" if quality["passed"] else "NOT READY"
    print(
        f"Pretrain quality gate: {state} | val={best_validation:.3f} "
        f"(need <= {args.max_validation_loss:.3f}), anchor={foundation_evaluation_loss:.3f} "
        f"(need <= {args.max_validation_loss:.3f}), healthy probes={healthy_fraction:.0%} "
        f"/ anchor={foundation_healthy_fraction:.0%} over {rollout_tokens} tokens "
        f"(need {args.min_healthy_probe_fraction:.0%}), "
        f"conditioning wins={current_conditioning['win_fraction']:.0%} / anchor={foundation_conditioning['win_fraction']:.0%} "
        f"(need {args.min_conditioning_win_fraction:.0%} on 32 cases with positive mean margin), "
        f"anchor regression={foundation_reference_regression:+.3f} "
        f"(allow <= {args.max_foundation_regression:.3f})."
    )
    print(f"Saved {stage} 10M student to {args.output}")


def main(stage: str = "pretrain") -> None:
    if stage not in {"pretrain", "anneal"}:
        raise ValueError(f"Unsupported plain-text stage {stage!r}.")
    is_anneal = stage == "anneal"
    parser = argparse.ArgumentParser(
        description=(
            "Anneal a quality-gated 10M LoopTransformer on selected high-quality plain text."
            if is_anneal
            else "Pretrain the 10M LoopTransformer on plain text."
        )
    )
    parser.add_argument(
        "--mixture-config",
        default=DEFAULT_MIXTURE_CONFIG,
        help="JSON list of weighted primary corpus specs (dataset, config, revision, text_field, splits, weight).",
    )
    parser.add_argument("--streaming", action="store_true", help="Read only bounded documents from a Hugging Face streaming split.")
    parser.add_argument("--streaming-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--streaming-max-source-tokens", type=int, default=4096)
    parser.add_argument(
        "--streaming-train-skip-documents",
        type=int,
        default=0,
        help="Skip this many raw train-stream records before shuffling; validation is unchanged.",
    )
    parser.add_argument("--max-documents", type=int, default=28_000)
    parser.add_argument("--max-validation-documents", type=int, default=50_000)
    parser.add_argument("--max-examples", type=int, default=200_000)
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        help="Fixed current-corpus validation-window budget; defaults to validation-fraction of max-examples.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5 if is_anneal else 3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100 if is_anneal else 500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-target-tokens", type=int, default=96)
    parser.add_argument(
        "--decoder-prompt-prefill-tokens",
        type=int,
        help="Opt-in count of final prompt tokens supplied to the decoder before generation; omitted resumes preserve checkpoint behavior.",
    )
    parser.add_argument(
        "--prefix-loss-weight",
        type=float,
        default=3.0,
        help="Extra training weight for rollout-critical first target tokens.",
    )
    parser.add_argument("--prefix-loss-tokens", type=int, default=1)
    parser.add_argument("--min-document-tokens", type=int, default=16)
    parser.add_argument(
        "--train-initial-prefix-span",
        type=int,
        default=1,
        help="Cycle initial training prefixes from 1 through this many tokens; validation stays fixed.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--probe-examples", type=int, default=16)
    parser.add_argument("--foundation-eval-examples", type=int, default=_DEFAULT_FOUNDATION_EVAL_EXAMPLES)
    parser.add_argument(
        "--rollout-tokens",
        type=int,
        default=64,
        help="Greedy continuation length used by the foundation readiness gate (matches play.py by default).",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=("validation", "rollout"),
        default="validation",
        help="Checkpoint ranking: lower validation CE (default) or bounded long-rollout repetition/health improvement.",
    )
    parser.add_argument(
        "--selection-rollout-tokens",
        type=int,
        help="Greedy length for --selection-strategy rollout; defaults to --rollout-tokens.",
    )
    parser.add_argument(
        "--selection-validation-tolerance",
        type=float,
        default=0.02,
        help="Allowed current and anchor CE increase when ranking anti-repetition candidates by rollout behavior.",
    )
    parser.add_argument(
        "--selection-max-repeat-regression",
        type=float,
        default=0.0,
        help="Allowed per-panel repeated-4-gram increase when using rollout checkpoint selection.",
    )
    parser.add_argument("--max-validation-loss", type=float, default=4.0)
    parser.add_argument("--min-healthy-probe-fraction", type=float, default=0.75)
    parser.add_argument(
        "--min-conditioning-win-fraction", type=float, default=10 / 32,
        help="Require strict true-prompt rank wins on 32 cases (default10/32; chance1/8), plus positive mean NLL margin.",
    )
    parser.add_argument(
        "--max-foundation-regression",
        type=float,
        default=0.1,
        help="Maximum allowed loss increase on the fixed foundation set during annealing.",
    )
    parser.add_argument("--thinking-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--init-checkpoint")
    if not is_anneal:
        parser.add_argument(
            "--architecture",
            choices=("compact-10m",),
            default=None,
            help="Fresh model architecture (default: compact-10m). Resumes keep their checkpoint config; mismatches fail.",
        )
    parser.add_argument("--tokenizer", help="Fresh tokenizer: compact-bpe (default), gpt2, or embedded-tokenizer JSON path. Resumes use checkpoint metadata.")
    parser.add_argument("--tokenizer-vocab-size", type=int, help="Vocabulary size for compact-bpe (default: 8192); checked strictly on resume.")
    parser.add_argument("--tokenizer-max-documents", type=int, default=10_000, help="Training-only document cap for fresh compact-bpe training.")
    parser.add_argument("--output", default=DEFAULT_ANNEAL_CHECKPOINT if is_anneal else DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    args.stage = stage
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"\n{error}\n")


if __name__ == "__main__":
    main()
