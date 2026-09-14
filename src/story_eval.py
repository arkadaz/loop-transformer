"""Independent, full-text behavior report for a TinyStories foundation checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.pretrain import _healthy_completion, _repeated_ngram_fraction
from src.pretrain_distill import checkpoint_metadata, load_checkpoint


DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
PANEL_SIZE = 32
PREFIX_TOKENS = 24
CUSTOM_PROMPTS = (
    ("lost_toy", "Ben lost his red toy car at the park. He looked under the bench, but it was not there."),
    ("found_toy", "Ben found his red toy car under the bench at the park. He smiled and held it tightly."),
    ("hungry_child", "Anna was very hungry after school. She opened her lunch box and saw a sandwich."),
    ("full_child", "Anna was full after eating her sandwich at school. She closed her lunch box and smiled."),
    ("rain", "Dark clouds came and rain began to fall on Mia's picnic."),
    ("sun", "The sun came out and shone warmly on Mia's picnic."),
    ("locked_door", "Tom tried the door, but it was locked. He could not get inside."),
    ("open_door", "Tom tried the door, and it was open. He could walk inside."),
    ("broken_bicycle", "Leo's bicycle was broken, so its wheel would not turn."),
    ("working_bicycle", "Leo's bicycle was working well, so its wheel turned smoothly."),
    ("frightened_dog", "The little dog was frightened by the loud thunder and hid under the table."),
    ("calm_dog", "The little dog was calm during the quiet evening and rested by the table."),
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_panel(tokenizer: Any) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, "default", split="validation", revision=REVISION)
    candidates = []
    for row in dataset:
        text = str(row["text"]).strip()
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= PREFIX_TOKENS + 8:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            candidates.append((digest, ids))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < PANEL_SIZE:
        raise RuntimeError(f"Only {len(candidates)} usable validation stories; need {PANEL_SIZE}.")
    return [
        {"id": digest, "prompt_token_ids": ids[:PREFIX_TOKENS], "prompt": tokenizer.decode(ids[:PREFIX_TOKENS])}
        for digest, ids in candidates[:PANEL_SIZE]
    ]


GREEDY_DECODING = {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}


def _completion(
    model, tokenizer: Any, prompt_ids: list[int], device: str, tokens: int, decoding: dict[str, Any] = GREEDY_DECODING
) -> dict[str, Any]:
    output = model.generate(
        torch.tensor([prompt_ids[-model.config.max_seq_len :]], device=device),
        thinking_effort="high",
        max_new_tokens=tokens,
        **decoding,
    )[0].tolist()[1:]
    eos = tokenizer.eos_token_id
    content = output[: output.index(eos)] if eos in output else output
    return {
        "token_ids": output,
        "text": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "generated_tokens": len(content),
        "ended_with_eos": eos in output,
        "healthy": _healthy_completion(output, eos),
        "repeated_4gram_fraction": _repeated_ngram_fraction(output, eos),
    }


def _run_panel(
    model, tokenizer: Any, panel: list[dict[str, Any]], device: str, tokens: int, decoding: dict[str, Any] = GREEDY_DECODING
) -> dict[str, Any]:
    rows = []
    for row in panel:
        result = _completion(model, tokenizer, row["prompt_token_ids"], device, tokens, decoding)
        rows.append({"id": row["id"], "prompt": row["prompt"], **result})
    healthy_count = sum(row["healthy"] for row in rows)
    mean_repeat = sum(row["repeated_4gram_fraction"] for row in rows) / len(rows)
    return {
        "max_new_tokens": tokens,
        "healthy_count": healthy_count,
        "healthy_fraction": healthy_count / len(rows),
        "mean_repeated_4gram_fraction": mean_repeat,
        "passes_mechanical_threshold": healthy_count >= 29 and mean_repeat <= 0.05,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen story-continuation behavior report; it does not train or select checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", help="JSON report path (default: next to checkpoint).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 keeps the frozen greedy report.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed; ignored for greedy decoding.")
    args = parser.parse_args()
    decoding = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    sampled = decoding != GREEDY_DECODING
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    metadata = checkpoint_metadata(checkpoint)
    stage = metadata.get("stage")
    if stage not in {"pretrain", "anneal"}:
        raise SystemExit(f"Expected a foundation checkpoint, got stage={stage!r}.")
    model, tokenizer = load_checkpoint(checkpoint, args.device)
    panel = _validation_panel(tokenizer)
    reports = [_run_panel(model, tokenizer, panel, args.device, tokens, decoding) for tokens in (64, 128)]
    custom = [
        {"name": name, "prompt": prompt,
         "at_64": _completion(model, tokenizer, tokenizer.encode(prompt, add_special_tokens=False), args.device, 64, decoding),
         "at_128": _completion(model, tokenizer, tokenizer.encode(prompt, add_special_tokens=False), args.device, 128, decoding)}
        for name, prompt in CUSTOM_PROMPTS
    ]
    mechanical_pass = all(report["passes_mechanical_threshold"] for report in reports)
    payload = {
        "version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_hash(checkpoint),
        "stage": stage,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "generation": {"thinking_effort": "high", "prefix_tokens": PREFIX_TOKENS, "seed": args.seed, **decoding},
        "mechanical_pass": mechanical_pass,
        "semantic_review_pass": None,
        "claim": (
            "Mechanical screen passed; manually review all custom contrasts before claiming simple story usability."
            if mechanical_pass else "Not usable for simple story continuation yet."
        ) + (" Measured with sampled decoding, not the greedy foundation gate." if sampled else ""),
        "validation_reports": reports,
        "custom_contrasts": custom,
    }
    default_suffix = ".story-eval.sampled.json" if sampled else ".story-eval.json"
    output = Path(args.output) if args.output else checkpoint.with_suffix(default_suffix)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Story evaluation ({'sampled' if sampled else 'greedy'}): mechanical={'PASS' if mechanical_pass else 'FAIL'}; "
        + "; ".join(f"{report['max_new_tokens']}t health={report['healthy_count']}/32 repeat4={report['mean_repeated_4gram_fraction']:.1%}" for report in reports)
    )
    print(f"Saved full prompts and outputs to {output}")


if __name__ == "__main__":
    main()
