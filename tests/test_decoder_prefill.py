import torch

from src import pretrain_distill
from src.model import LoopTransformer, LoopTransformerConfig


def tiny_model():
    return LoopTransformer(LoopTransformerConfig(
        vocab_size=32, d_model=16, n_heads=4, d_ff=32, num_decoder_layers=1,
        max_seq_len=16, decoder_start_token_id=1, eos_token_id=2,
    )).eval()


def test_training_and_generation_start_match_with_mixed_padding():
    torch.manual_seed(4)
    model = tiny_model()
    prompts = torch.tensor([[4, 5, 0, 0], [0, 6, 7, 8], [9, 0, 10, 0], [0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    starts = model.generate(prompts, input_attention_mask=mask, max_new_tokens=0)
    assert starts.tolist() == [[1]] * 4
    decoder_ids = torch.tensor([[1, 11, 12]] * 4)
    original = decoder_ids.clone()
    actual = model(prompts, decoder_ids, input_attention_mask=mask)
    hidden, _ = model.encode(prompts, attention_mask=mask)
    expected = model.decode(decoder_ids, hidden, model._encoder_mask(prompts, mask))
    assert torch.allclose(actual, expected)
    assert torch.equal(decoder_ids, original)  # Caller-owned batches are not mutated.
    next_ids = model.generate(prompts, input_attention_mask=mask, max_new_tokens=1)
    assert torch.equal(next_ids[:, 1], actual[:, 0].argmax(dim=-1))


def test_checkpoint_with_removed_config_fields_still_loads(tmp_path, monkeypatch):
    model = tiny_model()
    model.config.decoder_prompt_prefill_tokens = 3
    path = tmp_path / "legacy-fields.pt"
    pretrain_distill.save_checkpoint(path, model, ["test"], [], stage="pretrain")
    payload = torch.load(path, weights_only=True)
    payload["config"].update(embedding_dim=None, prompt_tail_decoder_start=False)
    torch.save(payload, path)
    marker = object()
    monkeypatch.setattr(pretrain_distill, "load_tokenizer", lambda: marker)
    restored, tokenizer = pretrain_distill.load_checkpoint(path)
    assert tokenizer is marker
    assert restored.config.decoder_prompt_prefill_tokens == 3
    prompts = torch.tensor([[4, 5]])
    assert torch.equal(restored.generate(prompts, max_new_tokens=2), model.generate(prompts, max_new_tokens=2))


def test_decoder_prompt_prefill_keeps_loss_alignment_and_hides_internal_tokens():
    torch.manual_seed(4)
    model = tiny_model()
    model.config.decoder_prompt_prefill_tokens = 3
    prompts = torch.tensor([[4, 5, 0, 0], [0, 6, 7, 8], [0, 0, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]])
    decoder_ids = torch.tensor([[1, 11, 12], [1, 13, 14], [1, 15, 16]])
    target_mask = torch.ones_like(decoder_ids, dtype=torch.bool)
    actual = model(
        prompts,
        decoder_ids,
        input_attention_mask=prompt_mask,
        target_attention_mask=target_mask,
    )
    prefix, prefix_mask = model._decoder_prompt_prefill(prompts, prompt_mask)
    hidden, _ = model.encode(prompts, attention_mask=prompt_mask)
    expected = model.decode(
        torch.cat([prefix, decoder_ids[:, 1:]], dim=1),
        hidden,
        model._encoder_mask(prompts, prompt_mask),
        torch.cat([prefix_mask, target_mask[:, 1:]], dim=1),
    )[:, 2:5]
    assert torch.allclose(actual, expected)
    generated = model.generate(prompts, input_attention_mask=prompt_mask, max_new_tokens=1)
    assert generated.shape == (3, 2)  # The three private prefill tokens are not returned.
    assert torch.equal(generated[:, 1], actual[:, 0].argmax(dim=-1))
