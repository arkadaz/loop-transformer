import torch

from src.model import KVCache, LoopTransformer, LoopTransformerConfig
from src.quant import TurboQuant, pack_bits, pack_codes, unpack_bits, unpack_codes


def small_model(prefill: int = 3) -> LoopTransformer:
    torch.manual_seed(0)
    config = LoopTransformerConfig(
        vocab_size=64, d_model=32, n_heads=4, d_ff=64, num_decoder_layers=2, max_seq_len=48,
        decoder_start_token_id=1, eos_token_id=2, decoder_prompt_prefill_tokens=prefill,
    )
    return LoopTransformer(config).eval()


def test_incremental_decode_matches_full_decode_including_a_multi_token_chunk():
    model = small_model()
    prompts = torch.randint(3, 60, (2, 7))
    encoder_mask = model._encoder_mask(prompts, None)
    encoder_hidden, _ = model.encode(prompts, num_loops=2)
    targets = torch.randint(3, 60, (2, 6))
    mask = torch.ones_like(targets, dtype=torch.bool)
    full = model.decode(targets, encoder_hidden, encoder_mask, mask)
    cache = KVCache(len(model.decoder))
    steps = [
        model.decode(targets[:, :4], encoder_hidden, encoder_mask, mask[:, :4], cache=cache),  # chunk: causal offset 0
        model.decode(targets[:, 4:5], encoder_hidden, encoder_mask, mask[:, :5], cache=cache),
        model.decode(targets[:, 5:6], encoder_hidden, encoder_mask, mask[:, :6], cache=cache),
    ]
    assert cache.length == 6
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_cached_generation_reproduces_uncached_generation_exactly():
    model = small_model()
    prompts = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    slow = model.generate(prompts, input_attention_mask=mask, max_new_tokens=12, use_cache=False, eos_token_id=63)
    fast = model.generate(prompts, input_attention_mask=mask, max_new_tokens=12, use_cache=True, eos_token_id=63)
    assert torch.equal(slow, fast)
    assert model.kv_cache.length == 3 + 12 - 1  # the last sampled token is never fed back


def test_turboquant_error_shrinks_with_bits_and_qjl_removes_score_bias():
    torch.manual_seed(0)
    k = torch.randn(2, 4, 256, 64)
    errors = []
    for bits in (2, 3, 4):
        tq = TurboQuant(64, bits, bits)
        keys = tq.quantize_keys(k)
        k_hat = tq._decode(keys.codes, keys.norms, tq.key_book, bits)
        errors.append(((k_hat - k).norm(dim=-1) / k.norm(dim=-1)).mean().item())
    assert errors[0] > errors[1] > errors[2] and errors[2] < 0.25

    tq = TurboQuant(64, 2, 2)
    keys = tq.quantize_keys(k)
    q = k + 0.3 * torch.randn_like(k)  # correlated queries expose the shrinkage bias of plain dequantisation
    exact = (q * k).sum(-1)
    plain = (q * tq._decode(keys.codes, keys.norms, tq.key_book, 2)).sum(-1)
    with_qjl = torch.diagonal(tq.key_scores(q, keys), dim1=-2, dim2=-1)
    scale = exact.abs().mean().item()
    assert (plain - exact).mean().item() < -0.05 * scale
    assert abs((with_qjl - exact).mean().item()) < 0.02 * scale
    assert tq.bits_per_coordinate() == (2 + 1 + 0.5, 2 + 0.25)


def test_pack_bits_roundtrip():
    bits = torch.rand(3, 5, 16) > 0.5
    packed = pack_bits(bits)
    assert packed.dtype == torch.uint8 and packed.shape == (3, 5, 2)
    assert torch.equal(unpack_bits(packed, 16), bits)
    codes = torch.randint(0, 8, (3, 5, 16))
    packed = pack_codes(codes, 3)
    assert packed.shape == (3, 5, 6) and torch.equal(unpack_codes(packed, 16, 3), codes)


def test_quantized_cache_generates_and_is_smaller_than_exact():
    model = small_model()
    prompts = torch.randint(3, 60, (2, 6))
    exact = model.generate(prompts, max_new_tokens=10, eos_token_id=63)
    exact_bytes = model.kv_cache.nbytes(self_attention_only=True)
    quantized = model.generate(prompts, max_new_tokens=10, kv_bits=3, eos_token_id=63)
    assert quantized.shape == exact.shape == (2, 11)
    assert model.kv_cache.nbytes(self_attention_only=True) < exact_bytes / 3
