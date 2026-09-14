"""KV-cache report: exactness of the cache, speed-up, TurboQuant fidelity per bit width, and memory per token."""
from __future__ import annotations

import argparse
import math
import time

import torch

from src.data import load_examples
from src.model import KVCache
from src.pretrain_distill import load_checkpoint, split_examples
from src.quant import TurboQuant


def batch(tokenizer, examples, device):
    ids = [tokenizer.encode(e["prompt"], add_special_tokens=False)[:128] for e in examples]
    width = max(map(len, ids))
    x = torch.tensor([i + [tokenizer.pad_token_id] * (width - len(i)) for i in ids], device=device)
    mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids], device=device)
    return x, mask


@torch.inference_mode()
def timed(model, x, mask, tokens, **kw):
    if x.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(x, input_attention_mask=mask, thinking_effort="high", max_new_tokens=tokens, **kw)
    if x.is_cuda:
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


@torch.inference_mode()
def forced_logits(model, x, mask, sequences, kv_bits):
    """Re-run the cached decoder over fixed token sequences and return the next-token logits at every step."""
    encoder_mask = model._encoder_mask(x, mask)
    encoder_hidden, _ = model.encode(x, attention_mask=mask, num_loops=model.resolve_loops("high"))
    prefix, prefix_mask = model._decoder_prompt_prefill(x, mask)
    head_dim = model.config.d_model // model.config.n_heads
    cache = KVCache(len(model.decoder), TurboQuant(head_dim, kv_bits, kv_bits, device=x.device) if kv_bits else None)
    tokens, token_mask, logits = prefix, prefix_mask, []
    pending = prefix
    for step in range(sequences.shape[1]):
        logits.append(model.decode(pending, encoder_hidden, encoder_mask, token_mask, cache=cache)[:, -1])
        pending = sequences[:, step : step + 1]
        tokens = torch.cat([tokens, pending], dim=1)
        token_mask = torch.cat([token_mask, torch.ones_like(pending, dtype=torch.bool)], dim=1)
    return torch.stack(logits, dim=1), cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the decoder KV cache and TurboQuant on real prompts.")
    parser.add_argument("--checkpoint", default="checkpoints/loop-transformer-10m-story-sft-05.pt")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, tokenizer = load_checkpoint(args.checkpoint, args.device)
    _, heldout = split_examples(load_examples(("tinystories_instruct_valid",), per_source=800, seed=7), 0.5, 7)
    x, mask = batch(tokenizer, heldout[: args.prompts], args.device)
    layers, head_dim, heads = len(model.decoder), model.config.d_model // model.config.n_heads, model.config.n_heads

    slow, slow_time = timed(model, x, mask, args.max_new_tokens, use_cache=False)
    fast, fast_time = timed(model, x, mask, args.max_new_tokens)
    same = sum(torch.equal(a, b) for a, b in zip(slow, fast))
    one_slow = timed(model, x[:1], mask[:1], args.max_new_tokens, use_cache=False)[1]
    one_fast = timed(model, x[:1], mask[:1], args.max_new_tokens)[1]
    print(f"Exactness: {same}/{len(fast)} greedy sequences identical with and without the cache.")
    print(f"Speed, {args.max_new_tokens} new tokens: batch {len(x)}: {slow_time:.1f}s -> {fast_time:.1f}s ({slow_time / fast_time:.1f}x); "
          f"batch 1: {one_slow:.1f}s -> {one_fast:.1f}s ({one_slow / one_fast:.1f}x).")

    sequences = fast[:, 1:]
    exact_logits, exact_cache = forced_logits(model, x, mask, sequences, 0)
    positions = layers * heads * exact_cache.length * len(x)
    print(f"\nSelf-attention KV memory per token per layer ({heads} heads x {head_dim} dims): "
          f"fp32 {exact_cache.nbytes(self_attention_only=True) / positions * heads:.0f} B (fp16 would be {heads * head_dim * 2 * 2:.0f} B)")
    print(f"{'cache':16s} {'bits K/V':>9s} {'bytes/tok/layer':>15s} {'mean |dlogit|':>13s} {'top-1 agree':>11s} {'KL(exact||q)':>12s}")
    for bits in (4, 3, 2, 1):
        logits, cache = forced_logits(model, x, mask, sequences, bits)
        delta = (logits - exact_logits).abs().mean().item()
        agree = (logits.argmax(-1) == exact_logits.argmax(-1)).float().mean().item()
        kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(exact_logits, -1), log_target=True, reduction="none").sum(-1).mean().item()
        kb, vb = TurboQuant(head_dim, bits, bits).bits_per_coordinate()
        print(f"TurboQuant {bits}-bit  {kb:4.1f}/{vb:<4.2f} {cache.nbytes(self_attention_only=True) / positions * heads:15.0f} {delta:13.4f} {agree:11.1%} {kl:12.4f}")
    print(f"\nExact logit scale for reference: mean |logit| = {exact_logits.abs().mean().item():.3f}. "
          f"Sequence length here: {exact_cache.length} decoder positions (prefill {model.config.decoder_prompt_prefill_tokens} + {sequences.shape[1]} generated). "
          f"Recompute cost without a cache grows with that length; the cached step is constant-size.")


if __name__ == "__main__":
    main()
