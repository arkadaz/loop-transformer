from types import SimpleNamespace

import pytest
import torch

from src.pretrain import _conditioning_panel_spec, _conditioning_ready, evaluate_prompt_conditioning
from src.pretrain_distill import PRETRAIN_QUALITY_GATE_VERSION, require_pretrain_ready


class Tokenizer:
    pad_token_id = 0

    def encode(self, *_args, **_kwargs):
        raise AssertionError("The evaluator must use exact stored IDs without re-tokenizing.")

    def decode(self, ids, **_kwargs):
        return " ".join(str(token) for token in ids)


def examples(count=32):
    return [
        {"prompt_token_ids": [i + 1] * 8, "target_token_ids": list(range(101 + i, 117 + i))}
        for i in range(count)
    ]


class ConditionalModel(torch.nn.Module):
    config = SimpleNamespace(max_seq_len=64, decoder_start_token_id=0)

    def __init__(self, *, tied=False):
        super().__init__()
        self.tied = tied

    def forward(self, input_ids, target_ids, *, input_attention_mask, target_attention_mask, thinking_effort):
        assert thinking_effort == "high"
        assert input_attention_mask.all() and target_attention_mask.all()
        assert target_ids.shape == (8, 8)
        assert target_ids[:, 0].eq(0).all()
        # All prompt alternatives must receive the same correctly shifted gold
        # prefix, with no extra EOS target at this deliberately open boundary.
        assert target_ids.eq(target_ids[:1]).all()
        assert torch.equal(target_ids[0, 1:], torch.arange(target_ids[0, 1], target_ids[0, 1] + 7))
        logits = torch.zeros((8, 8, 256))
        if not self.tied:
            predicted = input_ids[:, :1] + 100 + torch.arange(8)[None, :]
            logits.scatter_(2, predicted[..., None], 8.0)
        return logits


def test_conditioning_scores_exact_shifted_tokens_under_all_eight_prompts():
    model = ConditionalModel().train()
    result = evaluate_prompt_conditioning(model, Tokenizer(), examples(), "cpu")
    assert model.training
    assert result["case_count"] == 32
    assert result["win_fraction"] == 1
    assert result["mean_margin"] > 0
    assert _conditioning_ready(result, 10 / 32)


def test_conditioning_rejects_prompt_independent_ties():
    result = evaluate_prompt_conditioning(ConditionalModel(tied=True), Tokenizer(), examples(), "cpu")
    assert result["win_fraction"] == 0
    assert result["mean_margin"] == 0
    assert not _conditioning_ready(result, 0)


def test_conditioning_panel_is_frozen_and_distractors_are_unique():
    panel = _conditioning_panel_spec(examples(), Tokenizer())
    assert panel == _conditioning_panel_spec(examples(), Tokenizer())
    assert _conditioning_panel_spec([], Tokenizer(), stored_spec=panel) is panel
    for case in panel["cases"]:
        assert len(case["target_token_ids"]) == 8
        assert len({tuple(ids) for ids in [case["prompt_token_ids"], *case["distractors"]]}) == 8
    result = evaluate_prompt_conditioning(ConditionalModel(), Tokenizer(), [], "cpu", spec=panel)
    assert result["win_fraction"] == 1


def test_short_or_duplicate_panels_cannot_pass_even_with_perfect_scores():
    result = evaluate_prompt_conditioning(ConditionalModel(), Tokenizer(), examples(8), "cpu")
    assert result["win_fraction"] == 1
    assert not _conditioning_ready(result, 10 / 32)
    assert not _conditioning_panel_spec(examples(1) * 32, Tokenizer())["cases"]


def test_frozen_panel_rejects_duplicate_distractors():
    panel = _conditioning_panel_spec(examples(), Tokenizer())
    panel["cases"][0]["distractors"][0] = panel["cases"][0]["prompt_token_ids"]
    with pytest.raises(ValueError, match="conditioning case"):
        _conditioning_panel_spec([], Tokenizer(), stored_spec=panel)


def test_old_gate_claims_fail_and_word_overlap_no_longer_blocks_sft(tmp_path):
    path = tmp_path / "checkpoint.pt"
    quality = {
        "gate_version": PRETRAIN_QUALITY_GATE_VERSION - 1,
        "passed": True,
        "healthy_probes_ready": True,
        "foundation_healthy_probes_ready": True,
        "conditioning_ready": True,
        "foundation_conditioning_ready": True,
        "grounded_rollouts_ready": False,
        "foundation_grounded_rollouts_ready": False,
    }
    torch.save({"pretrain_quality": quality}, path)
    with pytest.raises(ValueError, match="quality gate"):
        require_pretrain_ready(path)
    quality["gate_version"] = PRETRAIN_QUALITY_GATE_VERSION
    torch.save({"pretrain_quality": quality}, path)
    require_pretrain_ready(path)
    quality["foundation_conditioning_ready"] = False
    torch.save({"pretrain_quality": quality}, path)
    with pytest.raises(ValueError, match="quality gate"):
        require_pretrain_ready(path)
