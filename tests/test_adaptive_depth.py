import torch

from src.model import LoopTransformer, LoopTransformerConfig


def model() -> LoopTransformer:
    torch.manual_seed(0)
    return LoopTransformer(LoopTransformerConfig(
        vocab_size=64, d_model=32, n_heads=4, d_ff=64, num_decoder_layers=1, max_seq_len=32,
        decoder_start_token_id=1, eos_token_id=2, decoder_prompt_prefill_tokens=2,
    )).eval()


def test_convergence_halting_bounds_match_fixed_depth():
    m = model()
    ids = torch.randint(3, 60, (3, 6))
    mask = torch.tensor([[1] * 6, [1] * 6, [1, 1, 1, 0, 0, 0]])
    never, _ = m.encode(ids, attention_mask=mask, num_loops=5, halt_threshold=0.0)
    used_never = m.loops_used.tolist()
    fixed, _ = m.encode(ids, attention_mask=mask, num_loops=5)
    assert torch.allclose(never, fixed) and used_never == [5, 5, 5]
    always, _ = m.encode(ids, attention_mask=mask, num_loops=5, halt_threshold=1e9)
    used_always = m.loops_used.tolist()
    one, _ = m.encode(ids, attention_mask=mask, num_loops=1)
    assert torch.allclose(always, one) and used_always == [1, 1, 1]


def test_halting_is_per_row_and_generate_accepts_it():
    m = model()
    ids = torch.randint(3, 60, (4, 5))
    _, history = m.encode(ids, num_loops=8, return_loop_history=True)
    changes = torch.stack([((history[k] - history[k - 1]).norm(dim=-1) / history[k].norm(dim=-1)).mean(1) for k in range(1, 9)], 1)
    threshold = changes[:, 3].median().item()  # about half the rows should stop by loop 4
    m.encode(ids, num_loops=8, halt_threshold=threshold)
    used = m.loops_used.tolist()
    assert min(used) < max(used) <= 8
    out = m.generate(ids, num_loops=8, halt_threshold=threshold, max_new_tokens=4, eos_token_id=63)
    assert out.shape == (4, 5) and m.loops_used.shape == (4,)
