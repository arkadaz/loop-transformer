"""Interact with a chosen LoopTransformer checkpoint without overstating its stage."""
import argparse
from pathlib import Path

import torch

import re

from src.pretrain_distill import PRETRAIN_QUALITY_GATE_VERSION, checkpoint_metadata, load_checkpoint
from src.rft import best_candidate


def input_context_limit(metadata: dict, config_max: int) -> tuple[int, bool]:
    """Use the recorded training input length without exceeding the architecture."""
    quality = metadata.get("pretrain_quality")
    recipe = quality.get("recipe") if isinstance(quality, dict) else None
    recorded = recipe.get("max_input_tokens") if isinstance(recipe, dict) else None
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded > 0:
        return min(recorded, config_max), True
    return config_max, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to inspect; pretraining checkpoints are continuers, not chat models.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; 0 means greedy decoding. 1.0 gives the most varied stories at no measured reward cost.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling mass; 1.0 disables it.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="1.0 disables the penalty.")
    parser.add_argument("--no-repeat-ngram", type=int, default=4, help="Block repeating any generated n-gram of this size; 0 disables.")
    parser.add_argument("--seed", type=int, help="Seed sampling for reproducible sessions.")
    parser.add_argument("--kv-bits", type=int, default=0, choices=(0, 1, 2, 3, 4), help="TurboQuant bits for the decoder KV cache; 0 = exact.")
    parser.add_argument("--no-kv-cache", action="store_true", help="Recompute the decoder each step (slow, for checks).")
    parser.add_argument("--max-loops", type=int, default=12, help="Loop cap for /effort auto.")
    parser.add_argument("--halt-threshold", type=float, default=0.1, help="/effort auto stops a row when its relative state change drops below this.")
    parser.add_argument("--best-of", type=int, default=1, help="Sample N stories and show the one the verifier scores highest (needs a sampling temperature).")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    temperature = args.temperature
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        metadata = checkpoint_metadata(Path(args.checkpoint))
        model, tokenizer = load_checkpoint(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Cannot load checkpoint: {error}")
    # Measured on the instruction task: fixed-depth checkpoints score best at 3 loops
    # (medium), not the 6 they trained at; depth-trained ones are flat from 1 to 8.
    effort = "medium"
    stage = str(metadata.get("stage") or "unknown")
    quality = metadata.get("pretrain_quality") or {}
    ready = quality.get("passed") and quality.get("gate_version") == PRETRAIN_QUALITY_GATE_VERSION
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    input_limit, recorded_context = input_context_limit(metadata, model.config.max_seq_len)
    context_status = (
        f"Trained input context limit={input_limit} tokens"
        if recorded_context else
        f"Trained input context unknown; architecture input limit={input_limit} tokens"
    )
    if stage in {"pretrain", "anneal"}:
        prompt_label, answer_label = "Story prefix", "Continuation"
        status = (
            "foundation gate passed: simple text continuation only; assistant behavior is not trained"
            if ready else "NOT READY: text continuation is not yet reliable"
        )
    else:
        prompt_label, answer_label = "You", "Model"
        status = (
            "instruction/evolution checkpoint; conversation quality is unverified"
            if stage in {"posttrain", "evolution"} else "checkpoint stage is unknown"
        )
    print(
        f"Loaded {args.checkpoint} on {device}: {parameter_count:,} parameters; stage={stage}; {status}. "
        f"{context_status}. "
        f"Decoder prompt prefill={model.config.decoder_prompt_prefill_tokens}. "
        f"Decoding: temperature={temperature:g} (0 = greedy), top_p={args.top_p:g}, "
        f"no_repeat_ngram={args.no_repeat_ngram}, repetition_penalty={args.repetition_penalty:g}; "
        f"KV cache {'off' if args.no_kv_cache else ('TurboQuant ' + str(args.kv_bits) + '-bit' if args.kv_bits else 'exact')}. "
        "Commands: /effort low|medium|high|auto, /temp <value>, /exit"
    )
    while True:
        try:
            user = input(f"{prompt_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in {"/exit", "/quit"}:
            break
        if not user:
            continue
        if user.startswith("/effort "):
            effort = user.split(maxsplit=1)[1]
            if effort not in {"low", "medium", "high", "auto"}:
                print("Use low, medium, high, or auto (stop when the encoder state stops changing).")
                effort = "medium"
            continue
        if user.startswith("/temp "):
            try:
                temperature = max(0.0, float(user.split(maxsplit=1)[1]))
            except ValueError:
                print("Use /temp <number>, for example /temp 0.7, or /temp 0 for greedy.")
            continue
        ids = tokenizer.encode(user, add_special_tokens=False)
        if len(ids) > input_limit:
            print(f"Input has {len(ids)} tokens; using the last {input_limit} tokens.")
            ids = ids[-input_limit:]
        input_ids = torch.tensor([ids], device=device)
        auto = effort == "auto"
        candidates = max(1, args.best_of if temperature > 0 else 1)
        output = model.generate(
            input_ids.repeat(candidates, 1),
            thinking_effort=None if auto else effort,
            num_loops=args.max_loops if auto else None,
            halt_threshold=args.halt_threshold if auto else None,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
            use_cache=not args.no_kv_cache,
            kv_bits=args.kv_bits,
        )
        texts = [tokenizer.decode(row[1:], skip_special_tokens=True).strip() for row in output]
        chosen, verdict = 0, ""
        if candidates > 1:  # verifier picks: required words from "Use the words: a, b, c", dialogue if requested
            words = re.search(r"[Uu]se the words?:\s*([^.\n]+)", user)
            example = {"required_words": [w.strip().lower() for w in words.group(1).split(",")] if words else [], "needs_dialogue": "dialogue" in user.lower()}
            best, reward = best_candidate(texts, [tokenizer.eos_token_id in row[1:].tolist() for row in output], example, min_reward=-1.0)
            chosen, verdict = best, f", best of {candidates} (verifier reward {reward:.2f})"
        answer = texts[chosen]
        decoding = ("greedy" if temperature == 0 else f"T={temperature:g} top_p={args.top_p:g}") + verdict
        loops = f"{int(model.loops_used[0])} loops used, cap {args.max_loops}" if auto else f"{model.resolve_loops(effort)} loops"
        print(f"{answer_label} ({effort}, {loops}, {decoding}): {answer}")


if __name__ == "__main__":
    main()
