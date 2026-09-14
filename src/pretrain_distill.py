"""Shared student training utilities and Gemma post-training stage."""
from __future__ import annotations

import argparse
import copy
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.data import DEFAULT_SOURCES, SOURCES, load_examples
from src.gemma_teacher import (
    GEMMA_TEACHER,
    TeacherAccessError,
    TeacherConfig,
    augment_with_gemma,
    ensure_teacher_access,
)
from src.model import LoopTransformer, LoopTransformerConfig
from src.student_tokenizer import tokenizer_from_payload


STUDENT_TOKENIZER = "gpt2"
CHECKPOINT_FORMAT_VERSION = 4
PRETRAIN_QUALITY_GATE_VERSION = 7
TRAINING_STATE_VERSION = 1
DEFAULT_PRETRAIN_CHECKPOINT = "checkpoints/loop-transformer-10m-pretrain.pt"
DEFAULT_ANNEAL_CHECKPOINT = "checkpoints/loop-transformer-10m-anneal.pt"
DEFAULT_POSTTRAIN_CHECKPOINT = "checkpoints/loop-transformer-10m-posttrain.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER)
    # The model itself enforces its 512-token limit. Pretraining tokenizes whole
    # source documents before selecting a window, so GPT-2's 1,024-token warning
    # is not a model-input error here.
    tokenizer.model_max_length = 1_000_000
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The student tokenizer needs PAD and EOS token IDs.")
    return tokenizer


def tokenize_target(tokenizer, text: str, limit: int, *, append_eos: bool = True) -> list[int]:
    """Tokenize a target, adding EOS only when the source really ends."""
    if limit < 2:
        raise ValueError("max_target_tokens must be at least 2.")
    tokens = tokenizer.encode(text, add_special_tokens=False)[:limit]
    # A full budget leaves no room for a genuine EOS; keep the content open.
    return [*tokens, tokenizer.eos_token_id] if append_eos and len(tokens) < limit else tokens


def _token_ids_from_example(example: dict[str, Any], field: str, tokenizer, limit: int) -> list[int]:
    """Use source token IDs when present so byte-level slices never round-trip through text."""
    ids = example.get(field)
    if ids is None:
        text_field = "prompt" if field == "prompt_token_ids" else "target"
        return tokenizer.encode(example[text_field], add_special_tokens=False)[:limit]
    if not isinstance(ids, (list, tuple)) or not all(isinstance(token, int) for token in ids):
        raise ValueError(f"{field} must be a sequence of integer token IDs.")
    return list(ids[:limit])


class TokenPairs(Dataset):
    def __init__(self, examples: Sequence[dict[str, Any]], tokenizer, max_input_tokens: int, max_target_tokens: int):
        if max_target_tokens < 2:
            raise ValueError("max_target_tokens must be at least 2.")
        self.items = []
        for example in examples:
            prompt = _token_ids_from_example(example, "prompt_token_ids", tokenizer, max_input_tokens)
            target = _token_ids_from_example(example, "target_token_ids", tokenizer, max_target_tokens)
            if example.get("target_eos", True) and len(target) < max_target_tokens:
                target.append(tokenizer.eos_token_id)
            if not prompt or not target:
                raise ValueError("Examples need at least one prompt and one target token.")
            self.items.append((prompt, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.items[index]


def collate_pairs(batch: list[tuple[list[int], list[int]]], pad_id: int, start_id: int) -> dict[str, torch.Tensor]:
    input_length = max(len(prompt) for prompt, _ in batch)
    target_length = max(len(target) for _, target in batch)
    input_ids = torch.full((len(batch), input_length), pad_id, dtype=torch.long)
    input_mask = torch.zeros((len(batch), input_length), dtype=torch.bool)
    labels = torch.full((len(batch), target_length), -100, dtype=torch.long)
    decoder_ids = torch.full((len(batch), target_length), pad_id, dtype=torch.long)
    target_mask = torch.zeros((len(batch), target_length), dtype=torch.bool)
    for index, (prompt, target) in enumerate(batch):
        input_ids[index, : len(prompt)] = torch.tensor(prompt)
        input_mask[index, : len(prompt)] = True
        labels[index, : len(target)] = torch.tensor(target)
        decoder_ids[index, 0] = start_id
        decoder_ids[index, 1 : len(target)] = torch.tensor(target[:-1])
        target_mask[index, : len(target)] = True
    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "decoder_ids": decoder_ids,
        "target_mask": target_mask,
        "labels": labels,
    }


def split_examples(examples: Sequence[dict[str, Any]], fraction: float = 0.1, seed: int = 42):
    """Make a reproducible, source-balanced held-out split before teacher augmentation."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(str(example["source"]), []).append(example)
    rng = random.Random(seed)
    train, validation = [], []
    for group in groups.values():
        group = list(group)
        rng.shuffle(group)
        validation_size = min(len(group) - 1, max(1, round(len(group) * fraction))) if len(group) > 1 else 0
        validation.extend(group[:validation_size])
        train.extend(group[validation_size:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if not train or not validation:
        raise ValueError("Need at least two usable examples from one selected source for a held-out validation split.")
    return train, validation


def _loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    prefix_loss_weight: float = 1.0,
    prefix_loss_tokens: int = 0,
) -> tuple[torch.Tensor, int, float]:
    """Cross-entropy with optional extra weight on rollout-critical target starts."""
    if prefix_loss_weight < 1 or prefix_loss_tokens < 0:
        raise ValueError("prefix_loss_weight must be >= 1 and prefix_loss_tokens cannot be negative.")
    valid = labels.ne(-100)
    tokens = int(valid.sum().item())
    weights = valid.to(logits.dtype)
    if prefix_loss_tokens and prefix_loss_weight != 1:
        weights[:, :prefix_loss_tokens] *= prefix_loss_weight
    per_token = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none")
    total = (per_token.view_as(labels) * weights).sum()
    normalizer = float(weights.sum().item())
    return total / max(normalizer, 1.0), tokens, normalizer


def _autocast(device: str):
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def cosine_learning_rate_scale(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    """Warm up, then smoothly decay a run's learning rate to min_lr_ratio."""
    if total_steps < 1 or warmup_steps < 0 or not 0 < min_lr_ratio <= 1:
        raise ValueError("Invalid learning-rate schedule settings.")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def _cpu_clone(value: Any) -> Any:
    """Detach optimizer data before placing it in an on-disk checkpoint."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _parameter_names(model: LoopTransformer) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def _training_state(
    model: LoopTransformer,
    optimizer: torch.optim.Optimizer,
    *,
    completed_optimizer_steps: int,
    supervised_tokens: int,
    completed_epochs: int,
    schedule: dict[str, Any],
) -> dict[str, Any]:
    """Capture state matching the current model weights, not merely the last batch."""
    return {
        "version": TRAINING_STATE_VERSION,
        "parameter_names": _parameter_names(model),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "completed_optimizer_steps": completed_optimizer_steps,
        "supervised_tokens": supervised_tokens,
        "completed_epochs": completed_epochs,
        # The cosine schedule intentionally restarts per command. Persist the
        # recipe so that checkpoint provenance states that clearly.
        "last_schedule": _cpu_clone(schedule),
    }


def _restore_optimizer_state(
    model: LoopTransformer,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    *,
    lr: float,
) -> tuple[int, int, int]:
    """Restore AdamW moments only when they match this exact parameter layout."""
    if state.get("version") != TRAINING_STATE_VERSION:
        raise ValueError("Unsupported training-state version in checkpoint.")
    if state.get("parameter_names") != _parameter_names(model):
        raise ValueError("Checkpoint optimizer state does not match this model's parameter layout.")
    optimizer_state = state.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint training state has no valid optimizer state.")
    try:
        optimizer.load_state_dict(optimizer_state)
    except (KeyError, ValueError, RuntimeError) as error:
        raise ValueError(f"Could not restore checkpoint optimizer state: {error}") from error
    # Optimizer state also serializes parameter-group settings. The current CLI
    # remains authoritative for a new continuation command.
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = 0.01
    try:
        steps = int(state.get("completed_optimizer_steps", 0))
        tokens = int(state.get("supervised_tokens", 0))
        epochs = int(state.get("completed_epochs", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint training progress counters must be integers.") from error
    if min(steps, tokens, epochs) < 0:
        raise ValueError("Checkpoint training progress counters cannot be negative.")
    return steps, tokens, epochs


@torch.inference_mode()
def evaluate(model: LoopTransformer, loader: DataLoader, device: str, *, description: str = "Validation") -> float:
    model.eval()
    loss_sum = token_count = 0
    for batch in tqdm(loader, desc=description, unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5):
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            logits = model(
                batch["input_ids"],
                batch["decoder_ids"],
                input_attention_mask=batch["input_mask"],
                target_attention_mask=batch["target_mask"],
                thinking_effort="high",
            )
            loss, count, _ = _loss(logits, batch["labels"])
        loss_sum += loss.item() * count
        token_count += count
    if not token_count:
        raise ValueError("Validation set is empty.")
    return loss_sum / token_count


@torch.inference_mode()
def evaluate_first_token_loss(
    model: LoopTransformer, loader: DataLoader, device: str, *, description: str = "First-token validation"
) -> float:
    """Measure the rollout-critical first decoder prediction separately from teacher-forced CE."""
    model.eval()
    loss_sum = token_count = 0
    for batch in tqdm(loader, desc=description, unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5):
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"][:, 0]
        valid = labels.ne(-100)
        if not valid.any():
            continue
        with _autocast(device):
            logits = model(
                batch["input_ids"],
                batch["decoder_ids"],
                input_attention_mask=batch["input_mask"],
                target_attention_mask=batch["target_mask"],
                thinking_effort="high",
            )[:, 0]
            loss_sum += F.cross_entropy(logits[valid], labels[valid], reduction="sum").item()
        token_count += int(valid.sum().item())
    if not token_count:
        raise ValueError("Validation set has no first target tokens.")
    return loss_sum / token_count


def evaluate_examples(
    model: LoopTransformer,
    examples: Sequence[dict[str, Any]],
    *,
    tokenizer,
    device: str,
    batch_size: int,
    max_input_tokens: int,
    max_target_tokens: int,
    description: str = "Validation",
    first_token_only: bool = False,
) -> float:
    """Evaluate a fixed serialized example set without constructing an optimizer."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    start_id = model.config.decoder_start_token_id
    if start_id is None:
        raise ValueError("Student model has no decoder start token.")
    dataset = TokenPairs(examples, tokenizer, max_input_tokens, max_target_tokens)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer.pad_token_id, start_id),
    )
    if first_token_only:
        return evaluate_first_token_loss(model, loader, device, description=description)
    return evaluate(model, loader, device, description=description)


def train(
    model: LoopTransformer,
    train_examples: Sequence[dict[str, Any]],
    validation_examples: Sequence[dict[str, Any]],
    *,
    tokenizer,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_input_tokens: int,
    max_target_tokens: int,
    seed: int = 42,
    checkpoint_every_steps: int = 0,
    on_checkpoint: Callable[[LoopTransformer, list[dict[str, float]], dict[str, Any]], None] | None = None,
    thinking_efforts: Sequence[str] = ("low", "medium", "high"),
    loop_range: tuple[int, int] | None = None,
    warmup_steps: int = 0,
    min_lr_ratio: float = 1.0,
    resume_training_state: dict[str, Any] | None = None,
    selection_callback: Callable[[LoopTransformer, float], tuple[bool, float, dict[str, float]]] | None = None,
    initial_selection_metric: float | None = None,
    prefix_loss_weight: float = 1.0,
    prefix_loss_tokens: int = 0,
) -> tuple[LoopTransformer, list[dict[str, float]]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps cannot be negative.")
    if max(max_input_tokens, max_target_tokens) > model.config.max_seq_len:
        raise ValueError("Token limits cannot exceed the model's max_seq_len.")
    prefill = model.config.decoder_prompt_prefill_tokens
    if prefill < 0 or prefill >= model.config.max_seq_len:
        raise ValueError("decoder_prompt_prefill_tokens must be in [0, max_seq_len).")
    if prefill and prefill + max_target_tokens - 1 > model.config.max_seq_len:
        raise ValueError("decoder prompt prefill plus target length exceeds max_seq_len.")
    start_id = model.config.decoder_start_token_id
    if start_id is None:
        raise ValueError("Student model has no decoder start token.")
    if not train_examples or not validation_examples:
        raise ValueError("Training and validation examples must both be non-empty.")
    if not thinking_efforts or any(effort not in {"low", "medium", "high"} for effort in thinking_efforts):
        raise ValueError("thinking_efforts must contain one or more of: low, medium, high.")
    if loop_range is not None and not 1 <= loop_range[0] <= loop_range[1] <= model.config.max_loop_steps:
        raise ValueError(f"loop_range must satisfy 1 <= lo <= hi <= {model.config.max_loop_steps}.")
    loop_rng = random.Random(seed + 7)  # variable-depth training: one random loop count per step
    if warmup_steps < 0 or not 0 < min_lr_ratio <= 1:
        raise ValueError("warmup_steps must be non-negative and min_lr_ratio must be in (0, 1].")
    if prefix_loss_weight < 1 or prefix_loss_tokens < 0:
        raise ValueError("prefix-loss-weight must be >= 1 and prefix-loss-tokens cannot be negative.")
    collate = lambda batch: collate_pairs(batch, tokenizer.pad_token_id, start_id)
    train_dataset = TokenPairs(train_examples, tokenizer, max_input_tokens, max_target_tokens)
    validation_dataset = TokenPairs(validation_examples, tokenizer, max_input_tokens, max_target_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    completed_steps = completed_tokens = completed_epochs = 0
    if resume_training_state is not None:
        completed_steps, completed_tokens, completed_epochs = _restore_optimizer_state(
            model,
            optimizer,
            resume_training_state,
            lr=lr,
        )
        print(
            f"Restored AdamW state: {completed_steps:,} completed optimizer steps / "
            f"{completed_tokens:,} supervised tokens."
        )
    total_steps = epochs * len(train_loader)
    schedule = {
        "type": "per_command_cosine",
        "base_lr": lr,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
        "command_steps": total_steps,
    }
    history: list[dict[str, float]] = []
    baseline_validation = evaluate(model, validation_loader, device, description="Baseline validation")
    print(f"baseline validation={baseline_validation:.4f}")
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_training_state = _training_state(
        model,
        optimizer,
        completed_optimizer_steps=completed_steps,
        supervised_tokens=completed_tokens,
        completed_epochs=completed_epochs,
        schedule=schedule,
    )
    # The initial checkpoint is always a recoverable candidate. A caller can
    # impose an additional constraint (for example, a fixed foundation loss)
    # before any later epoch replaces it.
    if initial_selection_metric is not None and not math.isfinite(initial_selection_metric):
        raise ValueError("initial_selection_metric must be finite when provided.")
    # Most runs rank checkpoints by validation CE.  Rollout-focused training
    # can supply a score for the untouched starting weights instead, so a
    # candidate must genuinely beat that baseline rather than merely be the
    # first allowed epoch.
    best_selection_metric = baseline_validation if initial_selection_metric is None else initial_selection_metric
    for epoch in range(epochs):
        model.train()
        loss_sum = token_count = 0
        objective_weight_sum = 0.0
        started = time.perf_counter()
        progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for step, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            global_step = epoch * len(train_loader) + step
            lr_scale = cosine_learning_rate_scale(global_step, total_steps, warmup_steps, min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_scale
            optimizer.zero_grad()
            thinking_effort = thinking_efforts[(epoch + step) % len(thinking_efforts)]
            with _autocast(device):
                logits = model(
                    batch["input_ids"],
                    batch["decoder_ids"],
                    input_attention_mask=batch["input_mask"],
                    target_attention_mask=batch["target_mask"],
                    thinking_effort=thinking_effort,
                    num_loops=loop_rng.randint(*loop_range) if loop_range else None,
                )
                loss, count, objective_weight = _loss(
                    logits,
                    batch["labels"],
                    prefix_loss_weight=prefix_loss_weight,
                    prefix_loss_tokens=prefix_loss_tokens,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * objective_weight
            objective_weight_sum += objective_weight
            token_count += count
            completed_steps += 1
            completed_tokens += count
            if checkpoint_every_steps and on_checkpoint and (step + 1) % checkpoint_every_steps == 0:
                on_checkpoint(
                    model,
                    history,
                    _training_state(
                        model,
                        optimizer,
                        completed_optimizer_steps=completed_steps,
                        supervised_tokens=completed_tokens,
                        completed_epochs=completed_epochs,
                        schedule=schedule,
                    ),
                )
                progress.write(f"checkpoint saved after {step + 1:,} batches")
            if step == 0 or (step + 1) % 25 == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                progress.set_postfix(
                    loss=f"{loss_sum / max(objective_weight_sum, 1):.3f}",
                    tok_s=f"{token_count / elapsed:,.0f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )
        validation_loss = evaluate(model, validation_loader, device, description=f"Validate epoch {epoch + 1}/{epochs}")
        completed_epochs += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(objective_weight_sum, 1),
            "validation_loss": validation_loss,
            "baseline_validation_loss": baseline_validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "cumulative_optimizer_steps": completed_steps,
            "cumulative_supervised_tokens": completed_tokens,
        }
        selection_allowed, selection_metric, selection_details = True, validation_loss, {}
        if selection_callback is not None:
            selection_allowed, selection_metric, selection_details = selection_callback(model, validation_loss)
            if not isinstance(selection_allowed, bool) or not isinstance(selection_metric, (int, float)):
                raise ValueError("selection_callback must return (bool, numeric metric, numeric details).")
            if not isinstance(selection_details, dict) or not all(
                isinstance(key, str) and isinstance(value, (int, float)) for key, value in selection_details.items()
            ):
                raise ValueError("selection_callback details must map strings to numbers.")
        record["selection_allowed"] = float(selection_allowed)
        record["selection_metric"] = float(selection_metric)
        record.update({f"selection_{key}": float(value) for key, value in selection_details.items()})
        history.append(record)
        # ``allowed`` only says a candidate did not violate a safety bound.
        # It must also beat the current best metric before its weights replace
        # the starting checkpoint.  Keep those two facts distinct in both the
        # history and terminal progress so a long run cannot look as though it
        # saved a candidate that was actually rolled back.
        selected = selection_allowed and selection_metric < best_selection_metric
        record["selection_selected"] = float(selected)
        if selected:
            decision = "selected"
        elif selection_allowed:
            decision = "allowed; retained prior best"
        else:
            decision = "rejected"
        print(
            f"epoch {epoch + 1}/{epochs}: train={record['train_loss']:.4f} val={validation_loss:.4f} "
            f"selection={decision} ({selection_metric:.4f})"
        )
        if selected:
            best_selection_metric = selection_metric
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_training_state = _training_state(
                model,
                optimizer,
                completed_optimizer_steps=completed_steps,
                supervised_tokens=completed_tokens,
                completed_epochs=completed_epochs,
                schedule=schedule,
            )
            if on_checkpoint:
                on_checkpoint(model, history, best_training_state)
    if best_state is not None:
        model.load_state_dict(best_state)
    # `best_training_state` is deliberately paired with `best_state`. A worse
    # final epoch must not cause the next command to combine old weights with
    # AdamW moments from different weights.
    model._loop_training_state = best_training_state
    return model, history


def save_checkpoint(
    path: str | Path,
    model: LoopTransformer,
    sources: Sequence[str],
    history: list[dict[str, float]],
    *,
    stage: str,
    parent_checkpoint: str | None = None,
    pretrain_quality: dict[str, Any] | None = None,
    foundation_anchor: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evolution_offset = getattr(model, "evolution_offset", None)
    checkpoint = {
        "state_dict": model.state_dict(),
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "config": asdict(model.config),
        "tokenizer": getattr(model, "_loop_tokenizer_payload", STUDENT_TOKENIZER),
        "teacher": GEMMA_TEACHER if stage in {"posttrain", "evolution"} else None,
        "sources": list(sources),
        "history": history,
        "parent_checkpoint": parent_checkpoint,
        "pretrain_quality": pretrain_quality,
        # Keep the fixed foundation identity even in a partial or rejected
        # checkpoint. Readiness is deliberately separate from this metadata.
        "foundation_anchor": _cpu_clone(foundation_anchor) if foundation_anchor is not None else None,
        "training_state": _cpu_clone(training_state) if training_state is not None else None,
        "evolution_offset": evolution_offset.detach().cpu() if isinstance(evolution_offset, torch.Tensor) else None,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata without constructing a tokenizer or model."""
    return torch.load(path, map_location="cpu", weights_only=True)


def require_pretrain_ready(path: str | Path) -> None:
    checkpoint = checkpoint_metadata(path)
    quality = checkpoint.get("pretrain_quality")
    if (
        not isinstance(quality, dict)
        or quality.get("gate_version") != PRETRAIN_QUALITY_GATE_VERSION
        or not quality.get("passed")
        or not quality.get("healthy_probes_ready")
        or not quality.get("foundation_healthy_probes_ready")
        or not quality.get("conditioning_ready")
        or not quality.get("foundation_conditioning_ready")
    ):
        raise ValueError(
            "The foundation quality gate has not passed. Continue plain-text foundation pretraining before annealing or Gemma SFT."
        )


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    *,
    expected_stage: str | Sequence[str] | None = None,
) -> tuple[LoopTransformer, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("This is an old checkpoint without architecture metadata. Train a new 10M checkpoint.")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("This checkpoint predates the current staged/evolution format. Train a new v4 checkpoint.")
    if expected_stage is not None:
        allowed_stages = {expected_stage} if isinstance(expected_stage, str) else set(expected_stage)
        if checkpoint.get("stage") not in allowed_stages:
            expected = " or ".join(sorted(allowed_stages))
            raise ValueError(f"Expected a {expected} checkpoint, got {checkpoint.get('stage')!r}.")
    tokenizer_spec = checkpoint.get("tokenizer", STUDENT_TOKENIZER)
    if isinstance(tokenizer_spec, dict):
        tokenizer = tokenizer_from_payload(tokenizer_spec)
        config = checkpoint["config"]
        if len(tokenizer) != config["vocab_size"] or tokenizer.eos_token_id != config["eos_token_id"]:
            raise ValueError("Embedded tokenizer does not match the model vocabulary or EOS ID.")
        if tokenizer.bos_token_id != config["decoder_start_token_id"]:
            raise ValueError("Embedded tokenizer does not match the decoder start ID.")
    elif tokenizer_spec == STUDENT_TOKENIZER:
        tokenizer = load_tokenizer()
    else:
        raise ValueError(f"Unsupported checkpoint tokenizer {tokenizer_spec!r}.")
    # Older checkpoints carry config fields from removed experiments; ignore them.
    known = {field.name for field in fields(LoopTransformerConfig)}
    model = LoopTransformer(LoopTransformerConfig(**{k: v for k, v in checkpoint["config"].items() if k in known})).to(device)
    model._loop_tokenizer_payload = tokenizer_spec
    model.load_state_dict(checkpoint["state_dict"])
    offset = checkpoint.get("evolution_offset")
    if offset is not None:
        expected = (model.config.num_latent_thoughts, model.config.d_model)
        if tuple(offset.shape) != expected:
            raise ValueError(f"Invalid evolution offset shape {tuple(offset.shape)}; expected {expected}.")
        model.evolution_offset = offset.to(device)
    return model.eval(), tokenizer


def _sft_foundation_guard(checkpoint, model, tokenizer, device, batch_size, limit_override=None):
    """Reuse the parent's fixed foundation limit when selecting SFT epochs."""
    # Imported here because foundation training itself imports these shared
    # utilities. No foundation data is rebuilt or sampled during SFT.
    from src.pretrain import _stored_foundation_anchor

    anchor = _stored_foundation_anchor(checkpoint)
    if anchor is None or anchor.get("reference_state") == "pending":
        raise ValueError("SFT retention requires a stored, trained foundation anchor.")
    quality = checkpoint.get("pretrain_quality") or {}
    recipe = quality.get("recipe") or {}
    recorded_limit = quality.get("max_foundation_regression", recipe.get("max_foundation_regression"))
    if limit_override is not None:
        recorded_limit = limit_override
    if recorded_limit is None:
        recorded_limit = 0.1
        print("Parent checkpoint has no recorded foundation regression limit; using the existing 0.1 default.")
    try:
        reference = float(anchor["reference_loss"])
        limit = float(recorded_limit)
        input_tokens = int(anchor["token_limits"]["max_input_tokens"])
        target_tokens = int(anchor["token_limits"]["max_target_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("SFT retention requires valid foundation reference and token limits.") from error
    if not math.isfinite(reference) or not math.isfinite(limit) or limit < 0:
        raise ValueError("SFT foundation reference and regression limit must be finite; limit must be non-negative.")
    if not 1 <= input_tokens <= model.config.max_seq_len or not 2 <= target_tokens <= model.config.max_seq_len:
        raise ValueError("Stored foundation token limits exceed this model's supported context.")

    def measure(current_model):
        return evaluate_examples(
            current_model, anchor["examples"], tokenizer=tokenizer, device=device,
            batch_size=min(batch_size, 64), max_input_tokens=input_tokens, max_target_tokens=target_tokens,
            description="SFT foundation retention",
        )

    baseline = measure(model)
    if not math.isfinite(baseline) or baseline - reference > limit:
        raise ValueError("The starting checkpoint already exceeds its immutable foundation regression limit.")
    print(f"SFT foundation guard: baseline={baseline:.4f}, immutable reference={reference:.4f}, allowed regression={limit:.4f}.")

    def select(current_model, validation_loss):
        loss = measure(current_model)
        regression = loss - reference
        start_regression = loss - baseline
        # Also preserve gains made since the original frozen reference. Both
        # limits stay fixed for the entire SFT run, including rejected epochs.
        accepted = math.isfinite(loss) and max(regression, start_regression) <= limit
        print(
            f"SFT foundation retention: loss={loss:.4f}, original regression={regression:+.4f}, "
            f"SFT-start regression={start_regression:+.4f}; {'accepted' if accepted else 'rejected'}."
        )
        return accepted, validation_loss, {
            "foundation_loss": loss,
            "foundation_reference_regression": regression,
            "foundation_start_regression": start_regression,
            "max_foundation_regression": limit,
        }

    return select, anchor


def run(args) -> None:
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.init_checkpoint, device, expected_stage=("pretrain", "anneal"))
    if args.allow_ungated:
        print("WARNING: --allow-ungated skips the foundation quality gate; this base has not passed its rollout screen.")
    else:
        require_pretrain_ready(args.init_checkpoint)
    foundation_selector, foundation_anchor = _sft_foundation_guard(
        checkpoint_metadata(args.init_checkpoint), model, tokenizer, device, args.batch_size,
        limit_override=args.max_foundation_regression,
    )
    ensure_teacher_access()
    source_names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    print(f"Loading up to {args.per_source} examples from: {', '.join(source_names)}")
    raw_examples = load_examples(source_names, per_source=args.per_source, seed=args.seed)
    train_raw, validation_raw = split_examples(raw_examples, args.validation_fraction, args.seed)
    print(f"Preparing {len(train_raw)} train and {len(validation_raw)} validation examples with {GEMMA_TEACHER} on {device}.")
    teacher_config = TeacherConfig(max_new_tokens=args.teacher_tokens, temperature=args.teacher_temperature)
    train_examples = augment_with_gemma(
        train_raw,
        cache_path=args.cache,
        config=teacher_config,
        device=device,
        progress_label="Gemma train targets",
        teacher_batch_size=args.teacher_batch_size,
    )
    validation_examples = augment_with_gemma(
        validation_raw,
        cache_path=args.cache,
        config=teacher_config,
        device=device,
        progress_label="Gemma validation targets",
        teacher_batch_size=args.teacher_batch_size,
    )
    print(f"Student parameters: {model.parameter_count:,}")
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
        selection_callback=foundation_selector,
        thinking_efforts=(args.thinking_effort,),  # match the foundation's loop count
        loop_range=tuple(args.loop_range) if args.loop_range else None,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    if not (getattr(model, "_loop_training_state", None) or {}).get("completed_optimizer_steps", 0):
        raise ValueError("No SFT epoch improved validation while retaining the foundation; starting weights were restored and no posttrain checkpoint was written.")
    save_checkpoint(
        args.output,
        model,
        source_names,
        history,
        stage="posttrain",
        parent_checkpoint=args.init_checkpoint,
        foundation_anchor=foundation_anchor,
    )
    print(f"Saved Gemma-post-trained 10M student to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train a pre-trained 10M LoopTransformer with Gemma 3 270M IT.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_SOURCES), help=f"Comma-separated sources: {', '.join(SOURCES)}")
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-target-tokens", type=int, default=96)
    parser.add_argument("--teacher-tokens", type=int, default=48)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--cache", default=".cache/gemma_targets.jsonl")
    parser.add_argument("--init-checkpoint", default=DEFAULT_PRETRAIN_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_POSTTRAIN_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--allow-ungated", action="store_true", help="Run SFT on a base that failed the foundation gate.")
    parser.add_argument("--max-foundation-regression", type=float, help="Override the stored foundation-loss regression limit.")
    parser.add_argument("--thinking-effort", choices=("low", "medium", "high"), default="high", help="Encoder loops used in SFT; keep the foundation setting.")
    parser.add_argument("--loop-range", type=int, nargs=2, metavar=("LO", "HI"), help="Random encoder loop count per step in [LO, HI]; makes representations depth-consistent.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.list_datasets:
        for name, source in SOURCES.items():
            print(f"{name}: {source.repository} ({source.license_note})")
        return
    try:
        run(args)
    except (TeacherAccessError, FileNotFoundError, ValueError) as error:
        parser.exit(2, f"\n{error}\n")


if __name__ == "__main__":
    main()
