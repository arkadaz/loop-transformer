"""Black-box evolutionary post-training over a small latent-thought offset."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from src.data import SOURCES, load_examples, story_checks
from src.pretrain_distill import DEFAULT_POSTTRAIN_CHECKPOINT, load_checkpoint, save_checkpoint, seed_everything, split_examples


DEFAULT_SOURCES = "gsm8k,arc_easy,arc_challenge,commonsenseqa,finance_sentiment"
_ANSWER_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)|[a-z]+")


def _normalise(text: str) -> str:
    """Keep signs and decimals: -5 and 5 must not receive the same reward."""
    return " ".join(_ANSWER_TOKEN.findall(text.lower()))


def _final_answer(text: str) -> str:
    """Score the SFT answer field rather than rewarding extra listed alternatives."""
    match = re.search(r"(?:^|\n)\s*final\s+answer\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def reward_for_answer(prediction: str, reference: str, *, ended: bool) -> float:
    """Reference-grounded reward; Gemma is not asked to overwrite factual labels."""
    predicted, target = _normalise(_final_answer(prediction)), _normalise(reference)
    match = 1.0 if target and predicted == target else 0.0
    tokens = predicted.split()
    repetition = max(Counter(tokens).values()) / len(tokens) if len(tokens) >= 4 else 0.0
    return match + (0.05 if ended else 0.0) - max(0.0, repetition - 0.5)


def reward_for_story(prediction: str, example: dict[str, Any], *, ended: bool) -> float:
    """Rule-verified reward: required words used, dialogue if asked, EOS bonus, repetition penalty."""
    words = prediction.lower().split()
    if len(words) < 20:
        return 0.0
    word_fraction, dialogue_ok = story_checks(prediction, example)
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    repetition = 1 - len(set(grams)) / len(grams)
    return word_fraction + (0.1 if dialogue_ok else 0.0) + (0.05 if ended else 0.0) - max(0.0, repetition - 0.05)


@torch.inference_mode()
def evaluate_offset(
    model,
    tokenizer: Any,
    examples: Sequence[dict[str, str]],
    offset: torch.Tensor,
    *,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    thinking_effort: str,
    batch_size: int = 32,
) -> float:
    model.evolution_offset = offset
    rewards = []
    encoded = [(e, tokenizer.encode(e["prompt"], add_special_tokens=False)[:max_input_tokens]) for e in examples]
    encoded = [(e, ids) for e, ids in encoded if ids]
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(ids) for _, ids in batch)
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (width - len(ids)) for _, ids in batch], device=device)
        mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for _, ids in batch], device=device)
        generated = model.generate(input_ids, input_attention_mask=mask, thinking_effort=thinking_effort, max_new_tokens=max_new_tokens)
        for (example, _), row in zip(batch, generated):
            tokens = row[1:].tolist()
            ended = tokenizer.eos_token_id in tokens
            prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if example.get("mode") == "story":
                rewards.append(reward_for_story(prediction, example, ended=ended))
            else:
                rewards.append(reward_for_answer(prediction, example["target"], ended=ended))
    return sum(rewards) / max(len(rewards), 1)


def run(args) -> None:
    if args.population < 4 or args.generations < 1 or args.elites < 1:
        raise ValueError("population must be >=4; generations and elites must be positive.")
    if args.elites >= args.population:
        raise ValueError("elites must be smaller than population.")
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(args.init_checkpoint, device, expected_stage="posttrain")
    source_names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    raw = load_examples(source_names, per_source=args.per_source, seed=args.seed)
    tuning, final_evaluation = split_examples(raw, args.validation_fraction, args.seed)
    tuning = tuning[: args.eval_examples]
    final_evaluation = final_evaluation[: args.eval_examples]
    if not tuning or not final_evaluation:
        raise ValueError("Need both tuning and held-out examples for evolution reward.")

    shape = (model.config.num_latent_thoughts, model.config.d_model)
    mean = getattr(model, "evolution_offset", torch.zeros(shape, device=device)).detach().clone().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    baseline_reward = evaluate_offset(
        model,
        tokenizer,
        tuning,
        mean,
        device=device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        thinking_effort=args.thinking_effort,
    )
    baseline_heldout_reward = evaluate_offset(
        model,
        tokenizer,
        final_evaluation,
        mean,
        device=device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        thinking_effort=args.thinking_effort,
    )
    best_offset, best_reward, history = mean.clone(), baseline_reward, []
    print(
        f"Evolving {mean.numel():,} latent-offset values on {len(tuning)} tuning examples using {device}. "
        f"Baseline tuning reward={baseline_reward:.3f}; held-out reward={baseline_heldout_reward:.3f}."
    )
    for generation in range(args.generations):
        half = args.population // 2
        noise = torch.randn((half, *shape), generator=generator, device=device)
        candidates = torch.cat([mean.unsqueeze(0) + args.sigma * noise, mean.unsqueeze(0) - args.sigma * noise], dim=0)
        if candidates.shape[0] < args.population:
            candidates = torch.cat([candidates, mean.unsqueeze(0)], dim=0)
        scores = torch.tensor(
            [
                evaluate_offset(
                    model,
                    tokenizer,
                    tuning,
                    candidate,
                    device=device,
                    max_input_tokens=args.max_input_tokens,
                    max_new_tokens=args.max_new_tokens,
                    thinking_effort=args.thinking_effort,
                )
                for candidate in tqdm(candidates, desc=f"evolution {generation + 1}/{args.generations}", unit="candidate")
            ],
            device=device,
        )
        elite_indices = scores.topk(args.elites).indices
        mean = candidates[elite_indices].mean(dim=0)
        generation_reward = scores[elite_indices[0]].item()
        if generation_reward > best_reward:
            best_reward, best_offset = generation_reward, candidates[elite_indices[0]].detach().clone()
        history.append({"generation": float(generation + 1), "best_reward": generation_reward, "mean_reward": scores.mean().item()})
        print(f"generation {generation + 1}/{args.generations}: best={generation_reward:.3f} mean={scores.mean().item():.3f}")
    model.evolution_offset = best_offset
    final_reward = evaluate_offset(
        model,
        tokenizer,
        final_evaluation,
        best_offset,
        device=device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        thinking_effort=args.thinking_effort,
    )
    history.append(
        {
            "baseline_tuning_reward": baseline_reward,
            "baseline_heldout_reward": baseline_heldout_reward,
            "best_tuning_reward": best_reward,
            "final_heldout_reward": final_reward,
        }
    )
    save_checkpoint(
        args.output,
        model,
        source_names,
        history,
        stage="evolution",
        parent_checkpoint=args.init_checkpoint,
    )
    print(
        f"Saved evolved checkpoint to {args.output}; best tuning reward={best_reward:.3f}, "
        f"held-out {baseline_heldout_reward:.3f} -> {final_reward:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve a small latent-thought offset after Gemma post-training.")
    parser.add_argument("--init-checkpoint", default=DEFAULT_POSTTRAIN_CHECKPOINT)
    parser.add_argument("--output", default="checkpoints/loop-transformer-10m-evolved.pt")
    parser.add_argument("--datasets", default=DEFAULT_SOURCES, help=f"Comma-separated sources: {', '.join(SOURCES)}")
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--thinking-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
