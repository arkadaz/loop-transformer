import copy

import pytest
import torch

from src import pretrain, pretrain_distill
from src.model import LoopTransformer, LoopTransformerConfig


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, **_kwargs):
        return [2 + ord(char) % 5 for char in text]


def guard(monkeypatch, student, health, losses):
    current = [{"prompt": "current", "target": "continuation"}]
    anchor = [{"prompt": "original", "target": "foundation"}]
    spec = {"case_indices": [0], "identity": "immutable"}
    original = copy.deepcopy(spec)
    health, losses = iter(health), iter(losses)
    monkeypatch.setattr(pretrain, "_grounded_rollout_spec", lambda *_args: {"identity": "current"})

    def rollouts(_model, _tokenizer, examples, _device, *, spec):
        assert examples is (anchor if spec["identity"] == "immutable" else current)
        return {"healthy_fraction": next(health)}

    def evaluate(_model, examples, **kwargs):
        assert examples is anchor
        assert kwargs["max_input_tokens"] == kwargs["max_target_tokens"] == 8
        return next(losses)

    monkeypatch.setattr(pretrain, "evaluate_grounded_rollouts", rollouts)
    monkeypatch.setattr(pretrain, "evaluate_examples", evaluate)
    result = pretrain._pretrain_selection_guard(
        student, Tokenizer(), "cpu", current, anchor, foundation_spec=spec,
        foundation_reference_loss=2.0, max_foundation_regression=0.1,
        batch_size=2, max_input_tokens=8, max_target_tokens=8,
    )
    assert spec == original
    return result


@pytest.mark.parametrize("candidate_health", [(0.25, 0.5), (0.5, 0.25)])
def test_either_panel_collapse_rejects_better_loss(monkeypatch, candidate_health):
    selector = guard(monkeypatch, None, [0.5, 0.5, *candidate_health], [1.9])
    allowed, loss, details = selector(None, 1.0)
    assert not allowed and loss == 1.0
    assert details["anchor_reference_regression"] < 0
    assert details["current_baseline_healthy_fraction"] == 0.5
    assert details["anchor_baseline_healthy_fraction"] == 0.5


def test_tolerance_allows_small_change_but_anchor_loss_still_blocks(monkeypatch):
    selector = guard(monkeypatch, None, [0.5, 0.5, 0.375, 0.375, 0.9, 0.9], [2.05, 2.2])
    assert selector(None, 1.5)[0]
    assert not selector(None, 1.0)[0]


def test_low_starting_health_can_improve_without_relaxing_readiness(monkeypatch):
    selector = guard(monkeypatch, None, [0.0, 0.0, 0.25, 0.25], [1.9])
    assert selector(None, 1.5)[0]
    captured = []
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr(pretrain, "run", captured.append)
    pretrain.main()
    assert captured[0].min_healthy_probe_fraction == 0.75
    assert captured[0].rollout_tokens == 64


def test_rollout_selection_keeps_safe_lower_repetition_candidate(monkeypatch):
    current = [{"prompt": "current", "target": "continuation"}]
    anchor = [{"prompt": "original", "target": "foundation"}]
    spec = {"case_indices": [0], "identity": "immutable"}
    rolls = iter([
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.7},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.7},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.5},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.5},
    ])
    losses = iter([1.0, 1.0, 1.01])  # current baseline, anchor baseline, candidate anchor
    calls = []
    monkeypatch.setattr(pretrain, "_grounded_rollout_spec", lambda *_args: {"identity": "current"})

    def rollouts(_model, _tokenizer, _examples, _device, *, spec, generated_tokens):
        calls.append((spec["identity"], generated_tokens))
        return next(rolls)

    monkeypatch.setattr(pretrain, "evaluate_grounded_rollouts", rollouts)
    monkeypatch.setattr(pretrain, "evaluate_examples", lambda *_args, **_kwargs: next(losses))
    selector = pretrain._pretrain_selection_guard(
        None, Tokenizer(), "cpu", current, anchor, foundation_spec=spec,
        foundation_reference_loss=1.0, max_foundation_regression=0.1,
        batch_size=2, max_input_tokens=8, max_target_tokens=8,
        rollout_tokens=64, selection_strategy="rollout", selection_validation_tolerance=0.02,
    )

    allowed, metric, details = selector(None, 1.01)

    assert calls == [("current", 64), ("immutable", 64), ("current", 64), ("immutable", 64)]
    assert allowed
    assert metric < selector.initial_selection_metric
    assert details["anchor_loss_regression"] == pytest.approx(0.01)


def test_rollout_selection_rejects_anchor_repeat_regression(monkeypatch):
    current = [{"prompt": "current", "target": "continuation"}]
    anchor = [{"prompt": "original", "target": "foundation"}]
    spec = {"case_indices": [0], "identity": "immutable"}
    rolls = iter([
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.7},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.7},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.5},
        {"healthy_fraction": 0.0, "mean_repeated_4gram_fraction": 0.8},
    ])
    losses = iter([1.0, 1.0, 1.0])
    monkeypatch.setattr(pretrain, "_grounded_rollout_spec", lambda *_args: {"identity": "current"})
    monkeypatch.setattr(
        pretrain, "evaluate_grounded_rollouts",
        lambda *_args, **_kwargs: next(rolls),
    )
    monkeypatch.setattr(pretrain, "evaluate_examples", lambda *_args, **_kwargs: next(losses))
    selector = pretrain._pretrain_selection_guard(
        None, Tokenizer(), "cpu", current, anchor, foundation_spec=spec,
        foundation_reference_loss=1.0, max_foundation_regression=0.1,
        batch_size=2, max_input_tokens=8, max_target_tokens=8,
        rollout_tokens=64, selection_strategy="rollout", selection_validation_tolerance=0.02,
    )

    allowed, _metric, details = selector(None, 1.0)

    assert not allowed
    assert details["anchor_repeat4_regression"] == pytest.approx(0.1)


def test_collapse_restores_best_accepted_weights_and_optimizer(monkeypatch):
    student = LoopTransformer(LoopTransformerConfig(
        vocab_size=8, d_model=8, n_heads=2, d_ff=16, num_decoder_layers=1,
        num_latent_thoughts=1, max_seq_len=16, decoder_start_token_id=1, eos_token_id=1,
    ))
    selector = guard(monkeypatch, student, [0.5, 0.5, 0.5, 0.5, 0.125, 0.125], [1.95, 1.9])
    validation_losses = iter([5.0, 4.0, 3.0])
    monkeypatch.setattr(pretrain_distill, "evaluate", lambda *_args, **_kwargs: next(validation_losses))
    selected = {}

    def capture(current_model, loss):
        result = selector(current_model, loss)
        if result[0]:
            selected.update({name: value.clone() for name, value in current_model.state_dict().items()})
        return result

    examples = [{"prompt": "ab", "target": "cd", "source": "unit"}]
    student, history = pretrain_distill.train(
        student, examples, examples, tokenizer=Tokenizer(), device="cpu",
        epochs=2, batch_size=1, lr=1e-3, max_input_tokens=8, max_target_tokens=8,
        selection_callback=capture,
    )
    assert [row["selection_allowed"] for row in history] == [1.0, 0.0]
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    assert all(torch.equal(value, selected[name]) for name, value in student.state_dict().items())
    assert student._loop_training_state["completed_optimizer_steps"] == 1
