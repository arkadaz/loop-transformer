"""Rejection-sampling fine-tuning data: sample N stories per training prompt and keep the best verifier-passing one.

The kept samples are the model's own successes (all required words, dialogue when asked, ended, no
repetition), so training on them teaches the behaviour in the model's own distribution. Feed the
output to ``sft`` as ``--datasets jsonl:<path>,...``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import load_examples
from src.evolution import reward_for_story
from src.pretrain_distill import load_checkpoint

FULL_PASS = 1.12  # every word (1.0) + dialogue ok (0.1) + ended (0.05) - repetition penalty: requires EOS and <= 8% repeated 4-grams


def best_candidate(texts: list[str], ended: list[bool], example: dict, min_reward: float = FULL_PASS) -> tuple[int | None, float]:
    """Index and reward of the best candidate, or (None, best) when none reaches ``min_reward``."""
    rewards = [reward_for_story(text, example, ended=done) for text, done in zip(texts, ended)]
    best = max(range(len(rewards)), key=rewards.__getitem__)
    return (best if rewards[best] >= min_reward else None), rewards[best]


@torch.inference_mode()
def sample_stories(model, tokenizer, examples, *, device, samples, loops, temperature, top_p, max_new_tokens, batch_size, seed):
    torch.manual_seed(seed)
    kept, best_rewards = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        ids = [tokenizer.encode(e["prompt"], add_special_tokens=False)[:128] for e in batch]
        width = max(map(len, ids))
        x = torch.tensor([i + [tokenizer.pad_token_id] * (width - len(i)) for i in ids], device=device)
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids], device=device)
        candidates = [[] for _ in batch]
        for _ in range(samples):
            out = model.generate(x, input_attention_mask=mask, num_loops=loops, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            for row_index, row in enumerate(out):
                tokens = row[1:].tolist()
                candidates[row_index].append((tokenizer.decode(tokens, skip_special_tokens=True).strip(), tokenizer.eos_token_id in tokens))
        for example, rows in zip(batch, candidates):
            index, reward = best_candidate([t for t, _ in rows], [d for _, d in rows], example)
            best_rewards.append(reward)
            if index is not None:
                kept.append({**example, "target": rows[index][0], "source": "rft", "mode": "anchor", "reward": reward})
        print(f"{min(start + batch_size, len(examples))}/{len(examples)} prompts, kept {len(kept)}", flush=True)
    return kept, best_rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rejection-sampling SFT data from a checkpoint's own verified stories.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="JSONL of kept (prompt, story) examples.")
    parser.add_argument("--per-source", type=int, default=12000, help="Instruct rows to draw prompts from (same seed as SFT).")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint, args.device)
    examples = load_examples(("tinystories_instruct",), per_source=args.per_source, seed=args.seed)
    kept, best = sample_stories(
        model, tokenizer, examples, device=args.device, samples=args.samples, loops=args.loops, temperature=args.temperature,
        top_p=args.top_p, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, seed=args.seed,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    print(f"kept {len(kept)}/{len(examples)} prompts ({len(kept) / len(examples):.0%}); mean best reward {sum(best) / len(best):.3f}; wrote {path}")


if __name__ == "__main__":
    main()
