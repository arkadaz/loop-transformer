import copy

import pytest
import torch

from src import pretrain_distill as training
from src.model import LoopTransformer, LoopTransformerConfig


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, **_kwargs):
        return [2 + ord(char) % 5 for char in text]


def model():
    return LoopTransformer(LoopTransformerConfig(
        vocab_size=8, d_model=8, n_heads=2, d_ff=16, num_decoder_layers=1,
        num_latent_thoughts=1, max_seq_len=16, decoder_start_token_id=1, eos_token_id=1,
    ))


def checkpoint(limit=0.1):
    return {
        "foundation_anchor": {
            "version": 3, "reference_state": "frozen", "reference_loss": 2.0,
            "examples": [{"prompt": "fixed", "target": "panel", "source": "foundation"}],
            "token_limits": {"max_input_tokens": 8, "max_target_tokens": 8},
            "conditioning_panel": {"frozen": "identity"},
        },
        "pretrain_quality": {"max_foundation_regression": limit},
    }


def test_guard_uses_recorded_immutable_limit_and_original_panel(monkeypatch):
    metadata = checkpoint(0.125)
    original = copy.deepcopy(metadata)
    losses = iter([2.02, 2.12, 2.14])

    def evaluate(_model, examples, **kwargs):
        assert examples is metadata["foundation_anchor"]["examples"]
        assert kwargs["max_input_tokens"] == kwargs["max_target_tokens"] == 8
        return next(losses)

    monkeypatch.setattr(training, "evaluate_examples", evaluate)
    student = model()
    selector, anchor = training._sft_foundation_guard(metadata, student, Tokenizer(), "cpu", 4)
    accepted, score, details = selector(student, 1.5)
    assert accepted and score == 1.5
    assert details["max_foundation_regression"] == 0.125
    assert selector(student, 1.0)[0] is False
    assert metadata == original
    assert anchor["conditioning_panel"] == original["foundation_anchor"]["conditioning_panel"]


def test_guard_older_metadata_uses_existing_default_and_rejects_stale_parent(monkeypatch, capsys):
    metadata = checkpoint()
    metadata["pretrain_quality"] = {}
    monkeypatch.setattr(training, "evaluate_examples", lambda *_args, **_kwargs: 2.05)
    selector, _ = training._sft_foundation_guard(metadata, model(), Tokenizer(), "cpu", 4)
    assert selector(model(), 1.0)[2]["max_foundation_regression"] == 0.1
    assert "existing 0.1 default" in capsys.readouterr().out
    monkeypatch.setattr(training, "evaluate_examples", lambda *_args, **_kwargs: 2.2)
    with pytest.raises(ValueError, match="already exceeds"):
        training._sft_foundation_guard(metadata, model(), Tokenizer(), "cpu", 4)


def test_better_sft_loss_cannot_replace_foundation_preserving_epoch(monkeypatch):
    student = model()
    foundation_losses = iter([2.0, 2.04, 2.2])
    sft_losses = iter([5.0, 4.0, 3.0])
    monkeypatch.setattr(training, "evaluate_examples", lambda *_args, **_kwargs: next(foundation_losses))
    monkeypatch.setattr(training, "evaluate", lambda *_args, **_kwargs: next(sft_losses))
    selector, _ = training._sft_foundation_guard(checkpoint(), student, Tokenizer(), "cpu", 2)
    selected_weights = None

    def capture_selection(current, validation_loss):
        nonlocal selected_weights
        result = selector(current, validation_loss)
        if result[0]:
            selected_weights = {name: value.clone() for name, value in current.state_dict().items()}
        return result

    examples = [{"prompt": "ab", "target": "cd", "source": "unit"}]
    student, history = training.train(
        student, examples, examples, tokenizer=Tokenizer(), device="cpu",
        epochs=2, batch_size=1, lr=1e-3, max_input_tokens=8, max_target_tokens=8,
        selection_callback=capture_selection,
    )
    assert [record["selection_allowed"] for record in history] == [1.0, 0.0]
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    assert all(torch.equal(value, selected_weights[name]) for name, value in student.state_dict().items())
    assert student._loop_training_state["completed_optimizer_steps"] == 1


def test_guard_also_retains_gains_since_the_original_foundation_reference(monkeypatch):
    losses = iter([1.5, 1.62])
    monkeypatch.setattr(training, "evaluate_examples", lambda *_args, **_kwargs: next(losses))
    student = model()
    selector, _ = training._sft_foundation_guard(checkpoint(), student, Tokenizer(), "cpu", 4)
    accepted, _, details = selector(student, 1.0)
    assert details["foundation_reference_regression"] < 0
    assert details["foundation_start_regression"] > 0.1
    assert not accepted


@pytest.mark.parametrize("limit", [-0.1, float("nan"), float("inf")])
def test_invalid_regression_limits_fail_before_training(limit):
    with pytest.raises(ValueError, match="finite"):
        training._sft_foundation_guard(checkpoint(limit), model(), Tokenizer(), "cpu", 4)
