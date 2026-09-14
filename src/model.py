"""Text-only recurrent encoder-decoder student model with an exact, optionally TurboQuant-compressed decoder KV cache."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quant import TurboQuant


@dataclass
class LoopTransformerConfig:
    vocab_size: int = 50_257
    d_model: int = 160
    n_heads: int = 4
    d_ff: int = 640
    num_encoder_loops: int = 3
    num_decoder_layers: int = 4
    num_latent_thoughts: int = 4
    max_seq_len: int = 512
    max_loop_steps: int = 16
    decoder_start_token_id: int | None = None
    eos_token_id: int | None = None
    # Optional decoder-side copy of the final prompt tokens. This is
    # parameter-free and reduces the first-token cross-attention bottleneck.
    decoder_prompt_prefill_tokens: int = 0


def make_compact_10m_config(
    vocab_size: int = 8192,
    *,
    decoder_start_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> LoopTransformerConfig:
    """A 10M student with a small vocabulary and wider transformer blocks."""
    if vocab_size != 8192:
        raise ValueError("compact-10m requires an 8192-token vocabulary.")
    return LoopTransformerConfig(
        vocab_size=vocab_size, d_model=256, n_heads=4, d_ff=1216,
        num_decoder_layers=6, decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        causal: bool = False,
        slot: "CacheSlot | None" = None,
    ) -> torch.Tensor:
        batch, query_len, width = x.shape
        context = x if context is None else context
        q = self._split(self.q(x))
        if slot is not None and slot.static and slot.filled:
            k, v = slot.k, slot.v  # encoder projections computed once per generation
        else:
            k, v = self._split(self.k(context)), self._split(self.v(context))
            if slot is not None and slot.static:
                slot.k, slot.v = k, v
        if slot is not None and not slot.static:
            scores, v = slot.append(q, k, v)  # growing cache; keys may be quantised, so the slot scores them
        else:
            scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        key_len = scores.shape[-1]

        if causal:  # new queries sit after any cached keys
            scores.masked_fill_(
                torch.triu(torch.ones(query_len, key_len, device=x.device, dtype=torch.bool), diagonal=key_len - query_len + 1),
                float("-inf"),
            )
        if key_mask is not None:
            if key_mask.shape != (batch, key_len):
                raise ValueError(f"Expected attention mask {(batch, key_len)}, got {tuple(key_mask.shape)}.")
            scores.masked_fill_(~key_mask.bool()[:, None, None, :], float("-inf"))

        output = F.softmax(scores, dim=-1) @ v
        return self.out(output.transpose(1, 2).contiguous().view(batch, query_len, width))


class CacheSlot:
    """K/V for one attention module. ``static`` slots hold encoder projections computed once;
    growing slots append per step and may hold TurboQuant-compressed keys and values."""

    def __init__(self, quantizer: TurboQuant | None = None, static: bool = False):
        self.quantizer, self.static = quantizer, static
        self.k = self.v = self.keys = self.values = None

    @property
    def filled(self) -> bool:
        return self.k is not None or self.keys is not None

    @property
    def length(self) -> int:
        if self.k is not None:
            return self.k.shape[2]
        return 0 if self.keys is None else self.keys.codes.shape[2]

    def append(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new keys/values; return (unscaled scores of q against every key, every value)."""
        if self.quantizer is None:
            self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
            self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
            return q @ self.k.transpose(-2, -1), self.v
        keys, values = self.quantizer.quantize_keys(k), self.quantizer.quantize_values(v)
        self.keys = keys if self.keys is None else self.keys.cat(keys)
        self.values = values if self.values is None else self.values.cat(values)
        return self.quantizer.key_scores(q, self.keys), self.quantizer.dequantize_values(self.values).to(q.dtype)

    def nbytes(self) -> int:
        if self.quantizer is None:
            return sum(t.numel() * t.element_size() for t in (self.k, self.v) if t is not None)
        return self.keys.nbytes() + self.values.nbytes() if self.keys is not None else 0


class KVCache:
    """Per decoder layer: a growing self-attention slot and a static cross-attention slot."""

    def __init__(self, num_layers: int, quantizer: TurboQuant | None = None):
        self.layers = [(CacheSlot(quantizer), CacheSlot(static=True)) for _ in range(num_layers)]

    @property
    def length(self) -> int:
        return self.layers[0][0].length

    def nbytes(self, *, self_attention_only: bool = False) -> int:
        return sum(s.nbytes() + (0 if self_attention_only else c.nbytes()) for s, c in self.layers)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.gelu(self.in_proj(x)))


class EncoderBlock(nn.Module):
    """A single shared block used repeatedly for latent recurrent depth."""
    def __init__(self, config: LoopTransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = Attention(config.d_model, config.n_heads)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.d_ff)
        self.loop_embedding = nn.Embedding(config.max_loop_steps, config.d_model)

    def forward(self, x: torch.Tensor, loop_index: int, mask: torch.Tensor) -> torch.Tensor:
        step = self.loop_embedding(torch.tensor(loop_index, device=x.device)).view(1, 1, -1)
        x = x + self.attention(self.norm1(x + step), key_mask=mask)
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: LoopTransformerConfig):
        super().__init__()
        self.self_norm = RMSNorm(config.d_model)
        self.self_attention = Attention(config.d_model, config.n_heads)
        self.cross_norm = RMSNorm(config.d_model)
        self.cross_attention = Attention(config.d_model, config.n_heads)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = FeedForward(config.d_model, config.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden: torch.Tensor,
        target_mask: torch.Tensor | None,
        encoder_mask: torch.Tensor,
        slots: tuple[CacheSlot, CacheSlot] | None = None,
    ) -> torch.Tensor:
        self_slot, cross_slot = slots if slots is not None else (None, None)
        x = x + self.self_attention(self.self_norm(x), key_mask=target_mask, causal=True, slot=self_slot)
        x = x + self.cross_attention(self.cross_norm(x), context=encoder_hidden, key_mask=encoder_mask, slot=cross_slot)
        return x + self.mlp(self.mlp_norm(x))


class LoopTransformer(nn.Module):
    """A compact text model with recurrent latent encoder depth."""
    def __init__(self, config: LoopTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.latent_thoughts = nn.Parameter(torch.randn(1, config.num_latent_thoughts, config.d_model) * 0.02)
        self.encoder = EncoderBlock(config)
        self.encoder_norm = RMSNorm(config.d_model)
        self.decoder = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_decoder_layers))
        self.decoder_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Use language-model-scale weights; Embedding defaults are far too large when tied."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def resolve_loops(self, thinking_effort: str | None = None, num_loops: int | None = None) -> int:
        loops = num_loops if num_loops is not None else {"low": 1, "medium": 3, "high": 6}.get(
            thinking_effort, self.config.num_encoder_loops
        )
        if not 1 <= loops <= self.config.max_loop_steps:
            raise ValueError(f"num_loops must be between 1 and {self.config.max_loop_steps}.")
        return loops

    def _encoder_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        latent_mask = torch.ones(
            input_ids.shape[0], self.config.num_latent_thoughts, dtype=torch.bool, device=input_ids.device
        )
        return torch.cat([latent_mask, attention_mask.bool()], dim=1)

    def _embed(self, token_ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        if start + token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Sequence exceeds max_seq_len={self.config.max_seq_len}.")
        positions = torch.arange(start, start + token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        return self.token_embedding(token_ids) + self.position_embedding(positions)

    def encode(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        num_loops: int | None = None,
        return_loop_history: bool = False,
        halt_threshold: float | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """With ``halt_threshold`` each row stops looping once its mean relative state change falls below
        it (``num_loops`` is then the cap); the loops actually run per row are left in ``self.loops_used``."""
        loops = self.resolve_loops(num_loops=num_loops)
        mask = self._encoder_mask(input_ids, attention_mask)
        thoughts = self.latent_thoughts
        offset = getattr(self, "evolution_offset", None)
        if offset is not None:
            if offset.shape != thoughts.shape[1:]:
                raise ValueError("evolution_offset must match [num_latent_thoughts, d_model].")
            thoughts = thoughts + offset.to(device=input_ids.device, dtype=thoughts.dtype).unsqueeze(0)
        hidden = torch.cat([thoughts.expand(input_ids.shape[0], -1, -1), self._embed(input_ids)], dim=1)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        history = [hidden.detach().clone()] if return_loop_history else None
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        done = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        self.loops_used = torch.full((hidden.shape[0],), loops, device=hidden.device) if halt_threshold is not None else None
        for loop_index in range(loops):
            updated = self.encoder(hidden, loop_index, mask) * valid
            if halt_threshold is None:
                hidden = updated
            else:  # adaptive depth: freeze rows whose state has stopped changing
                change = ((updated - hidden).norm(dim=-1) / updated.norm(dim=-1).clamp_min(1e-6) * mask).sum(1) / mask.sum(1)
                hidden = torch.where(done[:, None, None], hidden, updated)
                newly = ~done & (change < halt_threshold)
                self.loops_used[newly] = loop_index + 1
                done |= newly
            if history is not None:
                history.append(hidden.detach().clone())
            if done.all():
                break
        return self.encoder_norm(hidden), history

    def decode(
        self,
        target_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        """With a cache, ``target_ids`` are only the new positions and ``target_mask`` covers all cached ones."""
        hidden = self._embed(target_ids, start=cache.length if cache is not None else 0)
        for index, block in enumerate(self.decoder):
            hidden = block(hidden, encoder_hidden, target_mask, encoder_mask, slots=cache.layers[index] if cache is not None else None)
        return self.lm_head(self.decoder_norm(hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        input_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
        thinking_effort: str | None = None,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        loops = self.resolve_loops(thinking_effort, num_loops)
        encoder_mask = self._encoder_mask(input_ids, input_attention_mask)
        encoder_hidden, _ = self.encode(input_ids, attention_mask=input_attention_mask, num_loops=loops)
        prefill = self.config.decoder_prompt_prefill_tokens
        if prefill:
            prefix, prefix_mask = self._decoder_prompt_prefill(input_ids, input_attention_mask)
            decoder_ids = torch.cat([prefix, target_ids[:, 1:]], dim=1)
            if target_attention_mask is None:
                target_attention_mask = torch.ones_like(target_ids, dtype=torch.bool)
            decoder_mask = torch.cat([prefix_mask, target_attention_mask[:, 1:].bool()], dim=1)
            logits = self.decode(decoder_ids, encoder_hidden, encoder_mask, decoder_mask)
            # The final prefix token predicts target zero. Preserve the public
            # [batch, target_length, vocab] contract used by all losses.
            return logits[:, prefill - 1 : prefill - 1 + target_ids.shape[1]]
        return self.decode(target_ids, encoder_hidden, encoder_mask, target_attention_mask)

    def _decoder_start(self, input_ids: torch.Tensor, start_token_id: int | None = None) -> torch.Tensor:
        """One decoder start token per row."""
        start_id = self.config.decoder_start_token_id if start_token_id is None else start_token_id
        if start_id is None:
            raise ValueError("Configure decoder_start_token_id before decoding.")
        return input_ids.new_full((input_ids.shape[0], 1), start_id)

    def _decoder_prompt_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a fixed-width suffix of each valid prompt for the decoder.

        The final valid prompt token is always at the final prefill position,
        so it predicts target token zero. Short prompts are left-padded with
        the decoder-start token *as valid context*: masking those leading
        slots would create an all-masked causal-attention query and NaNs.
        """
        count = self.config.decoder_prompt_prefill_tokens
        if count < 1:
            raise ValueError("decoder_prompt_prefill_tokens must be positive when building a prefill.")
        if count >= self.config.max_seq_len:
            raise ValueError("decoder_prompt_prefill_tokens must be smaller than max_seq_len.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        starts = self._decoder_start(input_ids).squeeze(1)
        prefix = starts[:, None].expand(-1, count).clone()
        prefix_mask = torch.ones_like(prefix, dtype=torch.bool)
        for row in range(input_ids.shape[0]):
            tokens = input_ids[row][attention_mask[row].bool()][-count:]
            if tokens.numel():
                prefix[row, -tokens.numel() :] = tokens
        return prefix, prefix_mask

    @staticmethod
    def _sample_next_token(logits, context, *, temperature, top_p, repetition_penalty, no_repeat_ngram_size):
        """One next token per row. temperature 0 is greedy; the other knobs are inference-only constraints."""
        logits = logits.float().clone()
        if repetition_penalty != 1.0:
            seen = logits.gather(1, context)
            logits.scatter_(1, context, torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty))
        n = no_repeat_ngram_size
        if n and context.shape[1] >= n:
            for row, tokens in enumerate(context.tolist()):
                tail = tokens[len(tokens) - n + 1 :]
                banned = [tokens[i + n - 1] for i in range(len(tokens) - n + 1) if tokens[i : i + n - 1] == tail]
                if banned and len(set(banned)) < logits.shape[-1]:
                    logits[row, banned] = float("-inf")
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, order = probs.sort(dim=-1, descending=True)
            sorted_probs[sorted_probs.cumsum(-1) - sorted_probs > top_p] = 0
            probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        return torch.multinomial(probs, 1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        input_attention_mask: torch.Tensor | None = None,
        thinking_effort: str | None = None,
        num_loops: int | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        start_token_id: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        kv_bits: int = 0,
        halt_threshold: float | None = None,
    ) -> torch.Tensor:
        """``use_cache`` reproduces the uncached outputs exactly; ``kv_bits`` in 1..4 TurboQuant-compresses the
        self-attention cache (0 keeps it in full precision). The last cache is kept on ``self.kv_cache``."""
        if temperature < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0 or no_repeat_ngram_size < 0:
            raise ValueError("Use temperature >= 0, 0 < top_p <= 1, repetition_penalty > 0, no_repeat_ngram_size >= 0.")
        if kv_bits not in (0, 1, 2, 3, 4):
            raise ValueError("kv_bits must be 0 (exact) or 1..4.")
        loops = self.resolve_loops(thinking_effort, num_loops)
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if eos_token_id is None:
            raise ValueError("Configure decoder_start_token_id and eos_token_id before generation.")
        visible_start = self._decoder_start(input_ids, start_token_id)
        if self.config.decoder_prompt_prefill_tokens:
            generated, generated_mask = self._decoder_prompt_prefill(input_ids, input_attention_mask)
        else:
            generated = visible_start
            generated_mask = torch.ones_like(generated, dtype=torch.bool)
        if max_new_tokens > self.config.max_seq_len - generated.shape[1]:
            raise ValueError("Prompt prefill plus max_new_tokens exceeds max_seq_len.")

        encoder_mask = self._encoder_mask(input_ids, input_attention_mask)
        encoder_hidden, _ = self.encode(input_ids, attention_mask=input_attention_mask, num_loops=loops, halt_threshold=halt_threshold)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        returned = visible_start
        quantizer = TurboQuant(self.config.d_model // self.config.n_heads, kv_bits, kv_bits, device=input_ids.device) if kv_bits else None
        self.kv_cache = cache = KVCache(len(self.decoder), quantizer) if use_cache else None
        pending = generated  # tokens not yet in the cache: the whole prefill first, then one token per step
        for _ in range(max_new_tokens):
            logits = self.decode(
                pending if cache is not None else generated,
                encoder_hidden,
                encoder_mask,
                generated_mask,
                cache=cache,
            )[:, -1]
            next_token = self._sample_next_token(
                logits,
                generated,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            next_token = torch.where(finished[:, None], torch.full_like(next_token, eos_token_id), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            generated_mask = torch.cat([generated_mask, torch.ones_like(next_token, dtype=torch.bool)], dim=1)
            pending = next_token
            # Keep the long internal prefill private: callers always receive
            # [decoder start, generated tokens], as before this feature.
            returned = torch.cat([returned, next_token], dim=1)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break
        return returned
