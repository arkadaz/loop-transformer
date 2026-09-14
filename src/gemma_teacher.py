"""Gemma 3 270M IT is the project's only teacher."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from tqdm.auto import tqdm

from src.data import QA_PROMPT, qa_acceptable, story_checks


GEMMA_TEACHER = "google/gemma-3-270m-it"
GEMMA_MODEL_PAGE = f"https://huggingface.co/{GEMMA_TEACHER}"
PROMPT_VERSION = "answer-anchored-v1"


@dataclass(frozen=True)
class TeacherConfig:
    max_new_tokens: int = 48
    temperature: float = 0.0


class TeacherAccessError(RuntimeError):
    """Raised before data preparation when Gemma access is unavailable."""


def ensure_teacher_access() -> None:
    """Fail early with an actionable message instead of after dataset downloads."""
    from huggingface_hub import get_token, hf_hub_download

    instructions = (
        f"Gemma access is required. First accept the terms at {GEMMA_MODEL_PAGE}, then run "
        "`uv run hf auth login` with a Hugging Face read token. Verify it with `uv run hf auth whoami`."
    )
    if get_token() is None:
        raise TeacherAccessError(instructions)
    try:
        hf_hub_download(GEMMA_TEACHER, "config.json", token=True)
    except Exception as error:
        raise TeacherAccessError(instructions) from error


def _key(prompt: str, answer: str, config: TeacherConfig, mode: str = "") -> str:
    payload = json.dumps(
        {"teacher": GEMMA_TEACHER, "version": PROMPT_VERSION, "prompt": prompt, "answer": answer, "mode": mode, **asdict(config)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(path: Path) -> dict[str, tuple[str, bool]]:
    if not path.exists():
        return {}
    cache: dict[str, tuple[str, bool]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                item = json.loads(line)
                # Legacy records discarded the stopping reason. Unknown is
                # deliberately open, even when the text ends in punctuation.
                cache[str(item["key"])] = (str(item["response"]), item.get("ended") is True)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # Ignore an incomplete final line after an interrupted run.
    return cache


def _append_cache(path: Path, key: str, response: str, ended: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "response": response, "ended": ended}, ensure_ascii=False) + "\n")


STORY_REQUEST = (
    " Rules: use every listed word at least once; write about 100 words in simple language for a five-year-old;"
    " if dialogue is requested, include at least two lines in quotation marks; do not repeat sentences; reply with only the story."
)


def _clean_story(text: str) -> str:
    """Drop markdown marks and any title/preamble line before the story starts."""
    lines = [re.sub(r"[*#_]+", "", line).strip() for line in text.strip().splitlines()]
    while lines and (not lines[0] or len(lines[0].split()) < 6 or lines[0].endswith(":")):
        lines.pop(0)
    return "\n".join(line for line in lines if line)


QA_REQUEST = (
    "Read this story for young children. Write one simple question about what happens in it, then a short answer "
    "that uses words from the story. Reply with exactly two lines and nothing else:\nQuestion: ...\nAnswer: ...\n\nStory:\n"
)
_QA_PAIR = re.compile(r"Question:\s*(.+?)\s*Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _clean_line(text: str) -> str:
    return re.sub(r"[*#_`]+", "", text).strip().split("\n")[0].strip()


def _messages(prompt: str, answer: str, mode: str = "") -> list[dict[str, str]]:
    if mode == "qa":  # the teacher writes both the question and its answer about this passage
        return [{"role": "user", "content": QA_REQUEST + prompt.strip()}]
    if not answer.strip():  # story mode: the prompt is the whole instruction and Gemma writes the response
        return [{"role": "user", "content": prompt.strip() + STORY_REQUEST}]
    return [{
        "role": "user",
        "content": (
            "Write a short, factual explanation supporting the reference answer below. "
            "Do not change the answer, invent facts, or mention these instructions.\n\n"
            f"Task:\n{prompt.strip()}\n\nReference answer:\n{answer.strip()}"
        ),
    }]


def _useful_explanation(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return len(lowered) >= 8 and not any(
        phrase in lowered
        for phrase in ("no additional explanation", "please provide the reference answer", "i'm ready")
    )


def _teacher_stop_ids(model: Any, tokenizer: Any) -> list[int]:
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    configured = configured if isinstance(configured, (list, tuple)) else [configured]
    return list(dict.fromkeys(token for token in [*configured, tokenizer.eos_token_id] if token is not None))


def load_teacher(device: str, *, revision: str | None = None) -> tuple[Any, Any]:
    """Load Gemma once, frozen, with a device-appropriate dtype."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        load_kwargs = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_TEACHER, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(GEMMA_TEACHER, dtype=dtype, **load_kwargs).to(device).eval()
    except Exception as error:  # Depends on remote access and local VRAM.
        raise RuntimeError(
            "Could not load google/gemma-3-270m-it. Accept Gemma's Hugging Face terms, "
            "provide an HF token if needed, and verify CUDA/VRAM configuration."
        ) from error
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def generate_explanation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer: str,
    config: TeacherConfig,
    device: str,
) -> str:
    encoded = tokenizer.apply_chat_template(
        _messages(prompt, answer), add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        input_ids = encoded.to(device)
        attention_mask = torch.ones_like(input_ids)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": _teacher_stop_ids(model, tokenizer),
    }
    if config.temperature > 0:
        kwargs["temperature"] = config.temperature
    output = model.generate(**kwargs)
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_explanations(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    config: TeacherConfig,
    device: str,
    *,
    return_completion_flags: bool = False,
    modes: Sequence[str] | None = None,
) -> list[str] | list[tuple[str, bool]]:
    """Generate a batch of Gemma continuations, keeping cacheable answer order."""
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer.apply_chat_template(
            [_messages(prompt, answer, mode) for (prompt, answer), mode in zip(pairs, modes or [""] * len(pairs))],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    stop_ids = _teacher_stop_ids(model, tokenizer)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_ids,
    }
    if config.temperature > 0:
        kwargs["temperature"] = config.temperature
    output = model.generate(**kwargs)
    responses = []
    for row in output:
        continuation = row[input_ids.shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        ended = any(token in stop_ids for token in continuation.tolist())
        responses.append((text, ended) if return_completion_flags else text)
    return responses


def augment_with_gemma(
    examples: Sequence[Mapping[str, Any]],
    *,
    cache_path: str | Path,
    config: TeacherConfig = TeacherConfig(),
    device: str = "cpu",
    response_generator: Callable[[str, str], str | tuple[str, bool]] | None = None,
    progress_label: str = "Gemma targets",
    teacher_batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Anchor each target to its dataset answer and append a Gemma explanation."""
    if teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be positive.")
    path = Path(cache_path)
    cache = _read_cache(path)
    model = tokenizer = None
    # Replay examples keep their dataset target and never visit the teacher.
    augmented: list[dict[str, Any]] = [dict(example) for example in examples if example.get("mode") == "anchor"]
    examples = [example for example in examples if example.get("mode") != "anchor"]
    cache_hits = 0
    dropped = 0
    generated = 0
    progress = tqdm(total=len(examples), desc=progress_label, unit="example", dynamic_ncols=True)
    for start in range(0, len(examples), teacher_batch_size):
        batch = examples[start : start + teacher_batch_size]
        records: list[tuple[Mapping[str, Any], str, str, str, tuple[str, bool] | None]] = []
        missing: list[tuple[int, str, str, str]] = []
        for index, example in enumerate(batch):
            mode = str(example.get("mode", ""))
            prompt, answer = str(example["prompt"]), "" if mode in {"story", "qa"} else str(example["target"])
            key = _key(prompt, answer, config, mode)
            explanation = cache.get(key)
            if explanation is None:
                missing.append((index, prompt, answer, key))
            else:
                cache_hits += 1
            records.append((example, prompt, answer, key, explanation))
        if missing:
            if response_generator is not None:
                responses = [response_generator(prompt, answer) for _, prompt, answer, _ in missing]
            else:
                if model is None:
                    print(f"{progress_label}: loading Gemma on {device}...")
                    model, tokenizer = load_teacher(device)
                responses = generate_explanations(
                    model, tokenizer, [(prompt, answer) for _, prompt, answer, _ in missing], config, device,
                    return_completion_flags=True, modes=[str(batch[index].get("mode", "")) for index, _, _, _ in missing],
                )
            generated += len(missing)
            for (index, _prompt, _answer, key), response in zip(missing, responses, strict=True):
                response, ended = response if isinstance(response, tuple) else (response, False)
                response = response.strip() or "No additional explanation."
                cache[key] = (response, ended)
                _append_cache(path, key, response, ended)
                example, prompt, answer, _, _ = records[index]
                records[index] = (example, prompt, answer, key, (response, ended))
        for example, _prompt, answer, _key_value, response in records:
            explanation, ended = response or ("", False)
            item = dict(example)
            item["teacher"] = GEMMA_TEACHER
            if example.get("mode") == "qa":
                match = _QA_PAIR.search(explanation)
                question, written = (_clean_line(match.group(1)), _clean_line(match.group(2))) if match else ("", "")
                if not qa_acceptable(question, written, str(example["passage"])):
                    dropped += 1
                    continue
                item.update(
                    prompt=QA_PROMPT.format(passage=example["passage"], question=question),
                    target=written, reference_answer=written, question=question, mode="anchor", target_eos=True,
                )
                augmented.append(item)
                continue
            if example.get("mode") == "story":
                story = _clean_story(explanation)
                word_fraction, dialogue_ok = story_checks(story, example)
                accepted = ended and word_fraction == 1.0 and dialogue_ok and len(story.split()) >= 40
                item.update(teacher_story=story, teacher_accepted=accepted, target_eos=True)
                item["target"] = story if accepted else str(example["target"])  # dataset story is the anchor
                dropped += not accepted
                augmented.append(item)
                continue
            explanation = explanation.strip()
            if not _useful_explanation(explanation):
                explanation = ""
                dropped += 1
            item["reference_target"] = answer
            item["teacher_explanation"] = explanation
            item["target_eos"] = bool(example.get("target_eos", True)) and (not explanation or ended)
            item["target"] = f"Final answer: {answer.strip()}"
            if explanation:
                item["target"] += f"\n\nExplanation: {explanation}"
            augmented.append(item)
        progress.update(len(batch))
        progress.set_postfix(cached=cache_hits, generated=generated, refresh=False)
    progress.close()
    print(
        f"{progress_label}: ready ({cache_hits:,} cached, {generated:,} generated, {dropped:,} teacher outputs rejected)."
    )
    return augmented
