"""Instruction-stage report: verifier reward, constraint rates and output variety on held-out Instruct prompts.

This is the counterpart of ``story_eval`` for SFT/evolution checkpoints. It uses the same
prompts and split as the ``evolve`` command's held-out set, so numbers are comparable.
"""
from __future__ import annotations

import argparse

import torch

from src.data import load_examples, qa_checks, story_checks
from src.evolution import reward_for_qa, reward_for_story
from src.pretrain_distill import load_checkpoint, split_examples


@torch.inference_mode()
def score(model, tokenizer, prompts, *, device: str, max_new_tokens: int, **decoding) -> dict[str, float]:
    torch.manual_seed(0)
    rewards, words, all_three, dialogue, ended, openings, lily, loops = [], 0.0, 0, 0, 0, set(), 0, []
    for start in range(0, len(prompts), 32):
        batch = prompts[start : start + 32]
        ids = [tokenizer.encode(e["prompt"], add_special_tokens=False)[:128] for e in batch]
        width = max(map(len, ids))
        x = torch.tensor([i + [tokenizer.pad_token_id] * (width - len(i)) for i in ids], device=device)
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids], device=device)
        out = model.generate(x, input_attention_mask=mask, max_new_tokens=max_new_tokens, **{"thinking_effort": "high", **decoding})
        if getattr(model, "loops_used", None) is not None:
            loops.extend(model.loops_used.tolist())
        for example, row in zip(batch, out):
            tokens = row[1:].tolist()
            done = tokenizer.eos_token_id in tokens
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            fraction, dialogue_ok = story_checks(text, example)
            rewards.append(reward_for_story(text, example, ended=done))
            words += fraction
            all_three += fraction == 1.0
            dialogue += dialogue_ok
            ended += done
            openings.add(" ".join(text.split()[:4]).lower())
            lily += "lily" in text.lower()
    n = len(prompts)
    return {"reward": sum(rewards) / n, "words": words / n, "all3": all_three, "dialogue": dialogue, "ended": ended, "openings": len(openings), "lily": lily,
            "loops": sum(loops) / len(loops) if loops else None}


@torch.inference_mode()
def score_qa(model, tokenizer, prompts, *, device: str, max_new_tokens: int, **decoding) -> dict[str, float]:
    """Extractive Q&A: token F1 against the reference answer, exact match, grounding, answer length."""
    torch.manual_seed(0)
    rewards, f1s, exact, grounded, ended, lengths = [], 0.0, 0, 0.0, 0, 0
    for start in range(0, len(prompts), 32):
        batch = prompts[start : start + 32]
        ids = [tokenizer.encode(e["prompt"], add_special_tokens=False)[:128] for e in batch]
        width = max(map(len, ids))
        x = torch.tensor([i + [tokenizer.pad_token_id] * (width - len(i)) for i in ids], device=device)
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids], device=device)
        out = model.generate(x, input_attention_mask=mask, max_new_tokens=max_new_tokens, **{"thinking_effort": "high", **decoding})
        for example, row in zip(batch, out):
            tokens = row[1:].tolist()
            done = tokenizer.eos_token_id in tokens
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            f1, ground = qa_checks(text, example)
            rewards.append(reward_for_qa(text, example, ended=done))
            f1s, grounded, exact, ended, lengths = f1s + f1, grounded + ground, exact + (f1 == 1.0), ended + done, lengths + len(text.split())
    n = len(prompts)
    return {"reward": sum(rewards) / n, "f1": f1s / n, "exact": exact, "grounded": grounded / n, "ended": ended, "length": lengths / n,
            "loops": float(model.loops_used.float().mean()) if getattr(model, "loops_used", None) is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score instruction checkpoints on the evolve stage's held-out prompts.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--temperatures", default="1.0", help="Comma-separated sampling temperatures; greedy is always included.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7, help="Must match the evolve run's --seed for the same split.")
    parser.add_argument("--kv-bits", type=int, default=0, choices=(0, 1, 2, 3, 4), help="TurboQuant bits for the KV cache; 0 = exact.")
    parser.add_argument("--loops", default="", help="Comma-separated fixed loop counts to evaluate instead of high (6).")
    parser.add_argument("--halt-thresholds", default="", help="Comma-separated convergence thresholds for adaptive depth (cap --max-loops).")
    parser.add_argument("--max-loops", type=int, default=12)
    parser.add_argument("--task", choices=("story", "qa"), default="story")
    parser.add_argument("--qa-file", default="data/qa_valid.jsonl", help="Verified Q&A examples from src.qa_build (--task qa).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source = (f"jsonl:{args.qa_file}",) if args.task == "qa" else ("tinystories_instruct_valid",)
    raw = load_examples(source, per_source=800, seed=args.seed)
    _, heldout = split_examples(raw, 0.5, args.seed)
    prompts = heldout[: args.prompts]
    decodings = [("greedy", {})] + [
        (f"T={t}", {"temperature": float(t), "top_p": 0.9, "no_repeat_ngram_size": 4}) for t in args.temperatures.split(",") if t
    ]
    depths = [(f"{k} loops", {"thinking_effort": None, "num_loops": int(k)}) for k in args.loops.split(",") if k]
    depths += [(f"halt<{h}", {"thinking_effort": None, "num_loops": args.max_loops, "halt_threshold": float(h)}) for h in args.halt_thresholds.split(",") if h]
    if depths:  # depth sweep replaces the decoding sweep: greedy at each depth
        decodings = depths
    header = (f"{'reward':>7s} {'words':>6s} {'all3':>5s} {'dialog':>7s} {'ended':>6s} {'openings':>9s} {'Lily':>5s}" if args.task == "story"
              else f"{'reward':>7s} {'F1':>6s} {'exact':>6s} {'grounded':>9s} {'ended':>6s} {'words':>6s}")
    print(f"{len(prompts)} held-out {args.task} prompts")
    print(f"{'checkpoint':44s} {'setting':10s} {header} {'loops':>6s}")
    for path in args.checkpoints:
        model, tokenizer = load_checkpoint(path, args.device)
        if not hasattr(model, "evolution_offset"):
            model.evolution_offset = torch.zeros(model.config.num_latent_thoughts, model.config.d_model, device=args.device)
        for label, decoding in decodings:
            scorer = score if args.task == "story" else score_qa
            s = scorer(model, tokenizer, prompts, device=args.device, max_new_tokens=args.max_new_tokens, kv_bits=args.kv_bits, **decoding)
            name = path.replace("\\", "/").split("/")[-1][:44]
            used = f"{s['loops']:6.2f}" if s["loops"] is not None else "     -"
            body = (f"{s['reward']:7.3f} {s['words']:6.2f} {s['all3']:5d} {s['dialogue']:7d} {s['ended']:6d} {s['openings']:9d} {s['lily']:5d}" if args.task == "story"
                    else f"{s['reward']:7.3f} {s['f1']:6.2f} {s['exact']:6d} {s['grounded']:9.2f} {s['ended']:6d} {s['length']:6.1f}")
            print(f"{name:44s} {label:10s} {body} {used}")


if __name__ == "__main__":
    main()
