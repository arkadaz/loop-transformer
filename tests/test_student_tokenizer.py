import argparse
import json

import pytest
import torch

from src import pretrain, pretrain_distill
from src.model import LoopTransformer, LoopTransformerConfig, make_compact_10m_config
from src.student_tokenizer import (
    train_compact_tokenizer, tokenizer_from_payload, tokenizer_payload, validate_tokenizer_identity,
)


def tiny_tokenizer(text="A small training corpus with repeated words."):
    return train_compact_tokenizer([text] * 3, 280)


def test_byte_bpe_roundtrips_unseen_unicode_and_is_deterministic():
    tokenizer = tiny_tokenizer()
    text = "  café\nสวัสดี 中文 🦊\t\x00 e\u0301"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
    assert tokenizer_payload(tokenizer) == tokenizer_payload(tiny_tokenizer())
    assert len({tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}) == 3


def test_compact_checkpoint_is_portable_and_preserved_by_later_stages(tmp_path, monkeypatch):
    tokenizer = tiny_tokenizer()
    config = LoopTransformerConfig(
        vocab_size=len(tokenizer), d_model=16, n_heads=4, d_ff=32, num_decoder_layers=1,
        decoder_start_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    model = LoopTransformer(config).eval()
    model._loop_tokenizer_payload = tokenizer_payload(tokenizer)
    path = tmp_path / "portable.pt"
    pretrain_distill.save_checkpoint(path, model, ["test"], [], stage="pretrain")
    monkeypatch.setattr(pretrain_distill, "load_tokenizer", lambda: pytest.fail("Embedded tokenizer must load offline."))
    restored, loaded_tokenizer = pretrain_distill.load_checkpoint(path)
    validate_tokenizer_identity(tokenizer, loaded_tokenizer)
    ids = torch.tensor([tokenizer.encode("small corpus", add_special_tokens=False)])
    target = torch.tensor([[tokenizer.bos_token_id]])
    assert torch.equal(model(ids, target), restored(ids, target))
    for stage in ("anneal", "posttrain", "evolution"):
        path = tmp_path / f"{stage}.pt"
        pretrain_distill.save_checkpoint(path, restored, ["test"], [], stage=stage)
        restored, loaded_tokenizer = pretrain_distill.load_checkpoint(path, expected_stage=stage)
        validate_tokenizer_identity(tokenizer, loaded_tokenizer)


def test_identity_rejects_equal_size_different_vocabularies_and_tampering():
    first, second = tiny_tokenizer(), tiny_tokenizer("Other words differ completely from the original corpus.")
    assert len(first) == len(second)
    with pytest.raises(ValueError, match="differs"):
        validate_tokenizer_identity(first, second)
    payload = tokenizer_payload(first)
    payload["special_tokens"]["bos_token"] = "<|eos|>"
    with pytest.raises(ValueError, match="fingerprint"):
        tokenizer_from_payload(payload)


def test_resume_uses_embedded_tokenizer_and_checks_explicit_request(tmp_path):
    first, second = tiny_tokenizer(), tiny_tokenizer("Other words differ completely from the original corpus.")
    args = argparse.Namespace(tokenizer=None, tokenizer_vocab_size=None)
    pretrain._validate_resumed_tokenizer(args, first)
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(tokenizer_payload(first)), encoding="utf-8")
    args.tokenizer = str(path)
    pretrain._validate_resumed_tokenizer(args, first)
    with pytest.raises(ValueError, match="differs"):
        pretrain._validate_resumed_tokenizer(args, second)
    args.tokenizer = None
    args.tokenizer_vocab_size = len(first) + 1
    with pytest.raises(ValueError, match="vocab-size"):
        pretrain._validate_resumed_tokenizer(args, first)
    args.tokenizer = "compact-bpe"
    with pytest.raises(ValueError, match="Omit it"):
        pretrain._validate_resumed_tokenizer(args, first)


def test_tokenizer_training_uses_weighted_primary_training_partitions_only(monkeypatch):
    args = argparse.Namespace(tokenizer="compact-bpe", tokenizer_vocab_size=259, tokenizer_max_documents=12, max_documents=20)
    sources = [{"dataset": "one", "weight": 1}, {"dataset": "two", "weight": 2}]
    monkeypatch.setattr(pretrain, "_mixture_source_arguments", lambda args, source: argparse.Namespace(**vars(args), dataset=source["dataset"]))
    calls, observed = [], []

    def splits(sample_args):
        calls.append((sample_args.dataset, sample_args.max_documents))
        return [sample_args.dataset] * sample_args.max_documents, ["held-out secret"]

    def train(texts, vocab_size):
        observed.extend(texts)
        return list(range(vocab_size))

    monkeypatch.setattr(pretrain, "load_document_splits", splits)
    monkeypatch.setattr(pretrain, "train_compact_tokenizer", train)
    pretrain._fresh_tokenizer(args, sources)
    assert calls == [("one", 4), ("two", 8)]
    assert observed == ["one"] * 4 + ["two"] * 8


def test_compact_architecture_budget_and_resume_identity():
    model = LoopTransformer(make_compact_10m_config())
    assert model.parameter_count == 10_015_296
    pretrain._validate_resumed_architecture(model, "compact-10m")
    with pytest.raises(ValueError, match="conflicts"):
        pretrain._validate_resumed_architecture(LoopTransformer(LoopTransformerConfig(vocab_size=8192)), "compact-10m")
    with pytest.raises(ValueError, match="8192"):
        make_compact_10m_config(50_257)


def test_compact_pretrain_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--architecture", "compact-10m", "--tokenizer", "compact-bpe", "--tokenizer-vocab-size", "8192"])
    captured = []
    monkeypatch.setattr(pretrain, "run", captured.append)
    pretrain.main()
    assert captured[0].architecture == "compact-10m"
    assert captured[0].tokenizer == "compact-bpe"
    assert captured[0].tokenizer_vocab_size == 8192
