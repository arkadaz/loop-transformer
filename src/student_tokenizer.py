"""Portable, opt-in byte BPE for students with a small parameter budget."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = {"pad_token": "<|pad|>", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}


def train_compact_tokenizer(texts: Iterable[str], vocab_size: int = 8192):
    """Train on a caller-supplied, ordered training-only text stream."""
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("A byte BPE vocabulary needs at least 259 tokens.")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        texts,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS.values()),
            initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
            show_progress=True,
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, model_max_length=1_000_000, **SPECIAL_TOKENS)


def tokenizer_payload(tokenizer) -> dict:
    payload = {
        "kind": "byte-bpe",
        "version": 1,
        "backend_json": tokenizer.backend_tokenizer.to_str(),
        "special_tokens": {name: getattr(tokenizer, name) for name in SPECIAL_TOKENS},
    }
    payload["sha256"] = _fingerprint(payload)
    return payload


def _fingerprint(payload: Mapping) -> str:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    # Canonicalize the backend JSON too, making whitespace immaterial.
    content["backend_json"] = json.loads(content["backend_json"])
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def tokenizer_from_payload(payload: Mapping):
    if payload.get("kind") != "byte-bpe" or payload.get("version") != 1:
        raise ValueError("Unsupported embedded student tokenizer format.")
    if payload.get("sha256") != _fingerprint(payload):
        raise ValueError("The embedded tokenizer fingerprint does not match its contents.")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(payload["backend_json"]),
        model_max_length=1_000_000,
        **payload["special_tokens"],
    )
    if any(getattr(tokenizer, name + "_id") is None for name in SPECIAL_TOKENS):
        raise ValueError("Embedded student tokenizer needs PAD, BOS and EOS IDs.")
    return tokenizer


def validate_tokenizer_identity(actual, requested) -> None:
    """Token count alone cannot detect two vocabularies with different ID maps."""
    if tokenizer_payload(actual)["sha256"] != tokenizer_payload(requested)["sha256"]:
        raise ValueError("The requested tokenizer differs from the checkpoint tokenizer. Start a fresh model to change it.")
