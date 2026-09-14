import pytest

from src.pretrain import (
    FOUNDATION_ANCHOR_VERSION,
    _foundation_reference,
    _freeze_foundation_reference,
    _stored_foundation_anchor,
)
from src.pretrain_distill import checkpoint_metadata


def test_fresh_reference_freezes_selected_trained_loss_and_remains_immutable():
    panel = [{"prompt_token_ids": [1, 2], "target_token_ids": [3, 4]}]
    anchor = {
        "version": FOUNDATION_ANCHOR_VERSION,
        "examples": panel,
        "grounded_rollout": {"case_indices": [0]},
        **_foundation_reference(None, 10.8),
    }
    assert anchor["reference_state"] == "pending"
    assert anchor["reference_loss"] is None
    _freeze_foundation_reference(anchor, 10.8, 0, "random.pt")
    assert anchor["reference_state"] == "pending"
    _freeze_foundation_reference(anchor, 5.2, 3125, "trained.pt")
    assert anchor["reference_loss"] == 5.2
    assert anchor["reference_checkpoint"] == "trained.pt"

    reference = _foundation_reference(anchor, 5.29)
    descendant = {**anchor, **reference}
    _freeze_foundation_reference(descendant, 5.38, 6250, "continued.pt")
    assert descendant == anchor
    assert descendant["examples"] is panel
    assert 5.38 - descendant["reference_loss"] > 0.1


def test_pending_reference_survives_partial_checkpoint_and_freezes_on_completion(tmp_path):
    import torch

    anchor = {
        "version": FOUNDATION_ANCHOR_VERSION,
        "examples": [{"prompt": "a", "target": "b"}],
        **_foundation_reference(None, 10.8),
    }
    path = tmp_path / "partial.pt"
    torch.save({"foundation_anchor": anchor, "pretrain_quality": None}, path)
    restored = _stored_foundation_anchor(checkpoint_metadata(path))
    resumed = {**restored, **_foundation_reference(restored, 7.0)}
    assert resumed["reference_loss"] is None
    _freeze_foundation_reference(resumed, 5.0, 3125, "completed.pt")
    assert resumed["reference_loss"] == 5.0
    assert resumed["examples"] == anchor["examples"]


@pytest.mark.parametrize("reference", [
    {"reference_state": "frozen", "reference_loss": None},
    {"reference_state": "frozen", "reference_loss": float("nan")},
    {"reference_state": "pending", "reference_loss": 10.8},
    {"reference_loss": 4.0},
])
def test_invalid_v3_reference_cannot_silently_rebase(reference):
    with pytest.raises(ValueError):
        _foundation_reference({"version": FOUNDATION_ANCHOR_VERSION, **reference}, 5.0)
