import torch

from src.model import LoopTransformer, LoopTransformerConfig, make_compact_10m_config


def small_model() -> LoopTransformer:
    return LoopTransformer(
        LoopTransformerConfig(
            vocab_size=32,
            d_model=32,
            n_heads=4,
            d_ff=64,
            num_decoder_layers=1,
            max_seq_len=16,
            decoder_start_token_id=1,
            eos_token_id=2,
        )
    )


def test_forward_and_recurrent_history():
    model = small_model()
    inputs = torch.tensor([[3, 4, 5]])
    targets = torch.tensor([[1, 6, 7]])
    logits = model(inputs, targets, thinking_effort="high")
    _, history = model.encode(inputs, num_loops=4, return_loop_history=True)

    assert logits.shape == (1, 3, 32)
    assert len(history) == 5
    assert model.resolve_loops("low") == 1
    assert model.resolve_loops("high") == 6


def test_prompt_padding_is_masked():
    torch.manual_seed(0)
    model = small_model().eval()
    targets = torch.tensor([[1, 6]])
    mask = torch.tensor([[1, 1, 0, 0]])
    first = model(torch.tensor([[3, 4, 0, 0]]), targets, input_attention_mask=mask)
    second = model(torch.tensor([[3, 4, 9, 10]]), targets, input_attention_mask=mask)

    assert torch.allclose(first, second)


def test_10m_factory_is_about_ten_million_parameters():
    model = LoopTransformer(make_compact_10m_config(decoder_start_token_id=1, eos_token_id=2))
    assert 10_000_000 <= model.parameter_count <= 10_200_000


def test_language_model_weights_start_at_a_stable_scale_and_remain_tied():
    model = small_model()
    assert 0.015 < model.token_embedding.weight.std().item() < 0.025
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_generation_uses_configured_start_token():
    generated = small_model().generate(torch.tensor([[3, 4]]), max_new_tokens=0)
    assert generated.tolist() == [[1]]


def test_optional_evolution_offset_changes_latent_thoughts_without_new_parameters():
    model = small_model().eval()
    inputs, targets = torch.tensor([[3, 4]]), torch.tensor([[1, 6]])
    baseline = model(inputs, targets)
    model.evolution_offset = torch.ones(model.config.num_latent_thoughts, model.config.d_model)
    evolved = model(inputs, targets)
    assert model.parameter_count < 100_000
    assert not torch.allclose(baseline, evolved)


def test_generation_constraints_match_greedy_and_block_repeated_ngrams():
    torch.manual_seed(0)
    model = small_model().eval()
    prompt = torch.tensor([[3, 4, 5]])
    greedy = model.generate(prompt, max_new_tokens=8)
    # A vanishing nucleus keeps only the top token, so sampling must reproduce greedy decoding.
    nucleus = model.generate(prompt, max_new_tokens=8, temperature=0.5, top_p=1e-6)
    assert torch.equal(greedy, nucleus)

    constrained = model.generate(prompt, max_new_tokens=12, no_repeat_ngram_size=2)[0].tolist()
    content = constrained[: constrained.index(2)] if 2 in constrained else constrained
    bigrams = list(zip(content, content[1:]))
    assert len(bigrams) == len(set(bigrams))

    penalised = model.generate(prompt, max_new_tokens=8, repetition_penalty=1.5)
    assert penalised.shape == greedy.shape
