"""Build story-grounded Q&A training data: Gemma writes a question and answer per passage, a verifier keeps the good ones.

The answer must be short and made of words from the passage, so the student learns extraction from
its own context rather than recall it does not have. Output feeds ``sft`` and ``instruct_eval``
through ``jsonl:<path>``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import load_examples
from src.gemma_teacher import TeacherConfig, augment_with_gemma, ensure_teacher_access


def main() -> None:
    parser = argparse.ArgumentParser(description="Write verified story-grounded Q&A examples to JSONL.")
    parser.add_argument("--source", default="tinystories_qa", choices=("tinystories_qa", "tinystories_qa_valid"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=6000)
    parser.add_argument("--teacher-tokens", type=int, default=220, help="Keep this equal to the SFT run so the teacher cache is shared.")
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--cache", default=".cache/gemma_targets.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ensure_teacher_access()
    passages = load_examples((args.source,), per_source=args.per_source, seed=args.seed)
    examples = augment_with_gemma(
        passages,
        cache_path=args.cache,
        config=TeacherConfig(max_new_tokens=args.teacher_tokens, temperature=args.teacher_temperature),
        device=args.device,
        progress_label=f"Gemma {args.source}",
        teacher_batch_size=args.teacher_batch_size,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    print(f"kept {len(examples)}/{len(passages)} passages ({len(examples) / max(len(passages), 1):.0%}); wrote {path}")
    for row in examples[:3]:
        print(f"\nQ: {row['question']}\nA: {row['target']}\npassage: {row['passage'][:120]}...")


if __name__ == "__main__":
    main()
