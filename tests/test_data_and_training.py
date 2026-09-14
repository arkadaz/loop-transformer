from types import SimpleNamespace

import pytest
import torch

from src.data import arc, commonsense_qa, gsm8k
from src.evolution import reward_for_answer
from src.gemma_teacher import (
    TeacherAccessError,
    TeacherConfig,
    _read_cache,
    augment_with_gemma,
    ensure_teacher_access,
    generate_explanation,
    generate_explanations,
)
from src.model import LoopTransformer, LoopTransformerConfig
from src import pretrain, pretrain_distill
from src.pretrain import (
    _clean_document,
    _grounded_rollout_spec,
    _healthy_completion,
    _repeated_ngram_fraction,
    _row_documents,
    build_mixture_examples,
    evaluate_grounded_rollouts,
    load_document_splits,
    load_mixture_config,
    split_documents,
    text_windows,
)
from src.pretrain_distill import (
    CHECKPOINT_FORMAT_VERSION,
    PRETRAIN_QUALITY_GATE_VERSION,
    TokenPairs,
    checkpoint_metadata,
    cosine_learning_rate_scale,
    collate_pairs,
    load_checkpoint,
    require_pretrain_ready,
    save_checkpoint,
    split_examples,
    tokenize_target,
    train,
)


def test_curated_formatters_anchor_trusted_answers():
    math = gsm8k({"question": "What is 6 * 7?", "answer": "work #### 42"})
    choices = {"label": ["A", "B"], "text": ["wrong", "right"]}
    science = arc({"question": "Which?", "choices": choices, "answerKey": "B"}, "arc_easy")
    common = commonsense_qa({"question": "Which?", "choices": choices, "answerKey": "A"})

    assert math["target"] == "42"
    assert science["target"] == "right"
    assert common["target"] == "wrong"


def test_gemma_augmentation_caches_explanations_and_preserves_answer(tmp_path):
    examples = [{"prompt": "What is 2 + 2?", "target": "4", "source": "unit"}]
    cache = tmp_path / "targets.jsonl"
    result = augment_with_gemma(
        examples,
        cache_path=cache,
        config=TeacherConfig(max_new_tokens=8),
        response_generator=lambda prompt, answer: f"{prompt} supports {answer}.",
        progress_label="test",
    )
    cached = augment_with_gemma(
        examples,
        cache_path=cache,
        config=TeacherConfig(max_new_tokens=8),
        response_generator=lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")),
        progress_label="test",
    )

    assert result == cached
    assert result[0]["target"].startswith("Final answer: 4")
    assert cache.read_text(encoding="utf-8").count("\n") == 1


def test_incomplete_cache_line_does_not_block_resume(tmp_path):
    cache = tmp_path / "targets.jsonl"
    cache.write_text('{"key": "complete", "response": "ok"}\n{"key":', encoding="utf-8")
    assert _read_cache(cache) == {"complete": ("ok", False)}


def test_weak_teacher_explanations_do_not_become_training_targets(tmp_path):
    result = augment_with_gemma(
        [{"prompt": "Question", "target": "Answer", "source": "unit"}],
        cache_path=tmp_path / "targets.jsonl",
        response_generator=lambda *_: "No additional explanation.",
        progress_label="test",
    )
    assert result[0]["target"] == "Final answer: Answer"
    assert result[0]["teacher_explanation"] == ""


def test_target_tokenization_collation_and_balanced_split():
    class Tokenizer:
        eos_token_id = 9

        def encode(self, text, **_kwargs):
            return [ord(char) % 7 for char in text]

    assert tokenize_target(Tokenizer(), "abcd", 3) == [ord("a") % 7, ord("b") % 7, ord("c") % 7]
    assert tokenize_target(Tokenizer(), "abcd", 3, append_eos=False) == [ord("a") % 7, ord("b") % 7, ord("c") % 7]
    batch = collate_pairs([([1, 2], [3, 9]), ([4], [5, 6, 9])], pad_id=0, start_id=9)
    assert batch["labels"].tolist() == [[3, 9, -100], [5, 6, 9]]
    assert batch["decoder_ids"].tolist() == [[9, 3, 0], [9, 5, 6]]

    examples = [{"prompt": str(index), "target": "x", "source": "a" if index < 4 else "b"} for index in range(8)]
    train, validation = split_examples(examples, fraction=0.25)
    assert {item["source"] for item in validation} == {"a", "b"}
    assert len(train) + len(validation) == 8
    with pytest.raises(ValueError, match="held-out validation"):
        split_examples([{"prompt": "x", "target": "y", "source": "only"}])


def test_token_pairs_keep_exact_byte_level_window_ids():
    class Tokenizer:
        eos_token_id = 9

        def encode(self, _text, **_kwargs):
            return [99]

    pairs = TokenPairs(
        [
            {
                "prompt": "lossy replacement text",
                "target": "lossy replacement text",
                "prompt_token_ids": [1, 2, 3],
                "target_token_ids": [4, 5, 6],
                "target_eos": True,
            }
        ],
        Tokenizer(),
        max_input_tokens=2,
        max_target_tokens=3,
    )
    assert pairs[0] == ([1, 2], [4, 5, 6])


def test_teacher_generation_decodes_only_continuation():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def apply_chat_template(self, _messages, **_kwargs):
            return {"input_ids": torch.tensor([[4, 5]]), "attention_mask": torch.tensor([[1, 1]])}

        def decode(self, tokens, **_kwargs):
            assert tokens.tolist() == [7, 8]
            return "short explanation"

    class Teacher:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[4, 5, 7, 8]])

    answer = generate_explanation(Teacher(), Tokenizer(), "prompt", "answer", TeacherConfig(), "cpu")
    assert answer == "short explanation"


def test_batched_teacher_generation_keeps_continuations_and_padding_side():
    class Tokenizer:
        pad_token_id, eos_token_id, padding_side = 0, 1, "right"

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 2
            assert kwargs["padding"] is True
            assert self.padding_side == "left"
            return {"input_ids": torch.tensor([[0, 4, 5], [4, 5, 6]]), "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}

        def decode(self, tokens, **_kwargs):
            return "/".join(str(token) for token in tokens.tolist())

    class Teacher:
        def generate(self, **_kwargs):
            return torch.tensor([[0, 4, 5, 7], [4, 5, 6, 8]])

    tokenizer = Tokenizer()
    assert generate_explanations(Teacher(), tokenizer, [("a", "b"), ("c", "d")], TeacherConfig(), "cpu") == ["7", "8"]
    assert tokenizer.padding_side == "right"


def test_teacher_access_fails_before_network_without_a_token(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    with pytest.raises(TeacherAccessError, match="hf auth login"):
        ensure_teacher_access()


def test_batched_training_runs_on_a_small_model():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **_kwargs):
            return [2 + ord(character) % 10 for character in text]

    model = LoopTransformer(
        LoopTransformerConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            num_decoder_layers=1,
            max_seq_len=16,
            decoder_start_token_id=1,
            eos_token_id=1,
        )
    )
    examples = [
        {"prompt": "ab", "target": "cd", "source": "unit"},
        {"prompt": "ef", "target": "gh", "source": "unit"},
    ]
    _, history = train(
        model,
        examples,
        examples,
        tokenizer=Tokenizer(),
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_input_tokens=8,
        max_target_tokens=8,
    )

    assert len(history) == 1
    assert history[0]["validation_loss"] > 0


def test_foundation_continuation_keeps_adamw_state_matched_to_best_weights(monkeypatch, tmp_path):
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **_kwargs):
            return [2 + ord(character) % 10 for character in text]

    config = LoopTransformerConfig(
        vocab_size=16,
        d_model=16,
        n_heads=4,
        d_ff=32,
        num_decoder_layers=1,
        max_seq_len=16,
        decoder_start_token_id=1,
        eos_token_id=1,
    )
    examples = [
        {"prompt": "ab", "target": "cd", "source": "unit"},
        {"prompt": "ef", "target": "gh", "source": "unit"},
    ]
    # The second epoch is intentionally worse. The exported optimizer must
    # remain paired with the epoch-one model that train() restores.
    scores = iter([1.0, 0.5, 0.8])
    monkeypatch.setattr(pretrain_distill, "evaluate", lambda *_args, **_kwargs: next(scores))
    model, _ = train(
        LoopTransformer(config),
        examples,
        examples,
        tokenizer=Tokenizer(),
        device="cpu",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        max_input_tokens=8,
        max_target_tokens=8,
    )
    state = model._loop_training_state
    assert state["completed_optimizer_steps"] == 1
    assert state["completed_epochs"] == 1
    assert state["optimizer_state"]["state"]

    checkpoint = tmp_path / "foundation.pt"
    save_checkpoint(checkpoint, model, ["unit"], [], stage="pretrain", training_state=state)
    saved_state = checkpoint_metadata(checkpoint)["training_state"]
    assert saved_state["completed_optimizer_steps"] == 1

    # Continuing from the saved model and state increments the same AdamW
    # progress rather than creating a fresh optimizer.
    resumed = LoopTransformer(config)
    resumed.load_state_dict(model.state_dict())
    scores = iter([0.5, 0.4])
    monkeypatch.setattr(pretrain_distill, "evaluate", lambda *_args, **_kwargs: next(scores))
    resumed, _ = train(
        resumed,
        examples,
        examples,
        tokenizer=Tokenizer(),
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=2e-3,
        max_input_tokens=8,
        max_target_tokens=8,
        resume_training_state=saved_state,
    )
    resumed_state = resumed._loop_training_state
    assert resumed_state["completed_optimizer_steps"] == 2
    assert resumed_state["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(2e-3)


def test_cosine_schedule_warms_up_then_decays():
    assert cosine_learning_rate_scale(0, 10, 2, 0.1) == pytest.approx(0.5)
    assert cosine_learning_rate_scale(1, 10, 2, 0.1) == pytest.approx(1.0)
    assert cosine_learning_rate_scale(9, 10, 2, 0.1) < 0.2


def test_legacy_checkpoint_is_rejected_before_tokenizer_loading(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}, "config": {}}, path)
    with pytest.raises(ValueError, match="staged/evolution format"):
        load_checkpoint(path)


def test_checkpoint_stage_is_checked_before_tokenizer_loading(tmp_path):
    path = tmp_path / "posttrain.pt"
    torch.save({"state_dict": {}, "config": {}, "format_version": CHECKPOINT_FORMAT_VERSION, "stage": "posttrain"}, path)
    with pytest.raises(ValueError, match="Expected a pretrain checkpoint"):
        load_checkpoint(path, expected_stage="pretrain")


def test_plain_text_pretraining_windows_do_not_cross_documents():
    class Tokenizer:
        def encode(self, text, **_kwargs):
            return [ord(character) for character in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    examples = text_windows(["abcdef"], Tokenizer(), max_input_tokens=2, max_target_tokens=3, max_examples=10)
    assert [(item["prompt"], item["target"]) for item in examples] == [("a", "bc"), ("bc", "de"), ("de", "f")]
    assert [item["target_eos"] for item in examples] == [False, False, True]
    assert examples[0]["prompt_token_ids"] == [ord("a")]
    assert examples[0]["target_token_ids"] == [ord("b"), ord("c")]

    short = text_windows(["abc"], Tokenizer(), max_input_tokens=2, max_target_tokens=3, max_examples=10)
    assert [(item["prompt"], item["target"], item["target_eos"]) for item in short] == [("a", "bc", True)]
    train, validation = split_documents(["one", "two", "three"], 1 / 3, 42)
    assert len(train) == 2 and len(validation) == 1


def test_text_windows_can_vary_initial_training_prefixes():
    class Tokenizer:
        def encode(self, text, **_kwargs):
            return [ord(character) for character in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    examples = text_windows(
        ["abcdef", "ghijkl", "mnopqr"], Tokenizer(), max_input_tokens=2, max_target_tokens=3,
        max_examples=10, initial_prefix_span=3,
    )
    assert [examples[index]["prompt"] for index in (0, 3, 5)] == ["a", "gh", "o"]


def test_generic_document_loader_honours_text_field_and_keeps_code_structure():
    rows = [{"payload": {"body": "def add(x, y):\n    return x + y\n"}}]
    assert _row_documents(rows, "unit", "payload.body") == ["def add(x, y):\n    return x + y"]


def test_wiki40b_document_cleaning_removes_corpus_markup_but_keeps_text():
    raw = "_START_ARTICLE_ Ada Lovelace _START_SECTION_ Life _START_PARAGRAPH_ She wrote notes._NEWLINE_"
    assert _clean_document(raw, dataset="google/wiki40b") == "Ada Lovelace\nLife\nShe wrote notes."


def test_curated_primary_mixture_has_the_requested_50_30_sources():
    sources = load_mixture_config("configs/primary_fineweb_wiki40b.json")
    assert [(source["dataset"], source["config"], source["weight"]) for source in sources] == [
        ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", 50.0),
        ("google/wiki40b", "en", 30.0),
    ]
    assert sources[0]["use_fallback_validation"]
    assert sources[1]["validation_split"] == "validation"


def test_primary_mixture_allocates_and_interleaves_weighted_windows_deterministically(monkeypatch):
    class Tokenizer:
        def encode(self, text, **_kwargs):
            return [ord(character) for character in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    sources = [
        {
            "dataset": "unit/fineweb",
            "config": "dedup",
            "revision": "r1",
            "text_field": "text",
            "train_split": "train",
            "validation_split": None,
            "use_fallback_validation": True,
            "weight": 50.0,
        },
        {
            "dataset": "unit/wiki",
            "config": "en",
            "revision": "r2",
            "text_field": "text",
            "train_split": "train",
            "validation_split": "validation",
            "use_fallback_validation": False,
            "weight": 30.0,
        },
    ]

    def fake_splits(source_args):
        marker = "F" if source_args.dataset.endswith("fineweb") else "W"
        return [marker * 200], [marker.lower() * 200]

    monkeypatch.setattr(pretrain, "load_document_splits", fake_splits)
    args = SimpleNamespace(
        mixture_config="unit.json",
        max_input_tokens=2,
        max_target_tokens=2,
        min_document_tokens=1,
        seed=17,
    )
    train_a, validation_a, provenance = build_mixture_examples(
        args, Tokenizer(), sources, train_budget=80, validation_budget=16
    )
    train_b, validation_b, _ = build_mixture_examples(args, Tokenizer(), sources, train_budget=80, validation_budget=16)

    assert len(train_a) == 80 and len(validation_a) == 16
    assert sum(item["source"] == "plain_text:unit/fineweb/dedup" for item in train_a) == 50
    assert sum(item["source"] == "plain_text:unit/wiki/en" for item in train_a) == 30
    assert sum(item["source"] == "plain_text:unit/fineweb/dedup" for item in validation_a) == 10
    assert sum(item["source"] == "plain_text:unit/wiki/en" for item in validation_a) == 6
    assert [item["source"] for item in train_a] == [item["source"] for item in train_b]
    assert [item["source"] for item in validation_a] == [item["source"] for item in validation_b]
    assert [(source["train_windows"], source["validation_windows"]) for source in provenance["sources"]] == [(50, 10), (30, 6)]


def test_fallback_validation_membership_is_fixed_before_training_subset(monkeypatch):
    documents = [f"document-{index}" for index in range(20)]
    monkeypatch.setattr(pretrain, "_all_split_documents", lambda *_args: list(documents))

    def args(seed):
        return SimpleNamespace(
            dataset="unit",
            train_split="train",
            validation_split="",
            max_documents=8,
            max_validation_documents=20,
            validation_fraction=0.25,
            seed=seed,
        )

    train_a, validation_a = load_document_splits(args(1))
    train_b, validation_b = load_document_splits(args(2))
    assert validation_a == validation_b
    assert not (set(train_a) & set(validation_a))
    assert not (set(train_b) & set(validation_b))


def test_foundation_continuation_keeps_its_original_heldout_anchor():
    examples = [{"prompt": "old prompt", "target": "old target", "target_eos": True}]
    parent_anchor = {
        "examples": examples,
        "provenance": {"dataset": "Salesforce/wikitext", "dataset_revision": "main"},
    }
    fineweb_args = SimpleNamespace(
        dataset="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_revision="pinned-revision",
        text_field="text",
        validation_split="",
        foundation_eval_examples=64,
    )
    anchor, provenance = pretrain._foundation_anchor(
        parent_anchor,
        [{"prompt": "new prompt", "target": "new target", "target_eos": True}],
        fineweb_args,
    )
    assert anchor == examples
    assert provenance["dataset"] == "Salesforce/wikitext"


def test_partial_checkpoint_keeps_immutable_foundation_anchor(tmp_path):
    path = tmp_path / "partial.pt"
    anchor = {
        "version": 1,
        "examples": [{"prompt": "old", "target": "anchor", "target_eos": True}],
        "provenance": {"dataset": "unit"},
        "reference_loss": 2.5,
        "token_limits": {"max_input_tokens": 8, "max_target_tokens": 8},
        "replay_recipe": {"dataset": "unit", "dataset_config": "cfg", "dataset_revision": "r", "text_field": "text"},
    }
    torch.save({"pretrain_quality": None, "foundation_anchor": anchor}, path)
    restored = pretrain._stored_foundation_anchor(checkpoint_metadata(path))
    assert restored == anchor


def test_train_rolls_back_when_anchor_selector_rejects(monkeypatch):
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **_kwargs):
            return [2 + ord(character) % 10 for character in text]

    model = LoopTransformer(
        LoopTransformerConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            num_decoder_layers=1,
            max_seq_len=16,
            decoder_start_token_id=1,
            eos_token_id=1,
        )
    )
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    examples = [{"prompt": "ab", "target": "cd", "source": "unit"}, {"prompt": "ef", "target": "gh", "source": "unit"}]
    scores = iter([1.0, 0.5])
    monkeypatch.setattr(pretrain_distill, "evaluate", lambda *_args, **_kwargs: next(scores))
    model, history = train(
        model,
        examples,
        examples,
        tokenizer=Tokenizer(),
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_input_tokens=8,
        max_target_tokens=8,
        selection_callback=lambda *_args: (False, 0.5, {"anchor_loss": 9.0}),
    )
    assert history[0]["selection_allowed"] == 0.0
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())


def test_train_reports_an_allowed_candidate_that_does_not_beat_the_start(monkeypatch):
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **_kwargs):
            return [2 + ord(character) % 10 for character in text]

    model = LoopTransformer(
        LoopTransformerConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            d_ff=32,
            num_decoder_layers=1,
            max_seq_len=16,
            decoder_start_token_id=1,
            eos_token_id=1,
        )
    )
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    examples = [{"prompt": "ab", "target": "cd", "source": "unit"}, {"prompt": "ef", "target": "gh", "source": "unit"}]
    scores = iter([1.0, 0.5])
    monkeypatch.setattr(pretrain_distill, "evaluate", lambda *_args, **_kwargs: next(scores))
    model, history = train(
        model,
        examples,
        examples,
        tokenizer=Tokenizer(),
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        max_input_tokens=8,
        max_target_tokens=8,
        selection_callback=lambda *_args: (True, 2.0, {"anchor_loss": 0.5}),
        initial_selection_metric=1.0,
    )
    assert history[0]["selection_allowed"] == 1.0
    assert history[0]["selection_selected"] == 0.0
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())


def test_streaming_loader_is_bounded_and_has_a_stable_disjoint_holdout(monkeypatch):
    rows = [
        {"id": f"row-{index}", "payload": {"body": f"document {index}"}, "token_count": 20}
        for index in range(100)
    ]
    rows.insert(0, {"id": "too-long", "payload": {"body": "never keep me"}, "token_count": 9_999})

    class Stream:
        def __init__(self, values):
            self.values = values

        def shuffle(self, *, seed, buffer_size):
            assert buffer_size == 10
            offset = seed % len(self.values)
            return Stream(self.values[offset:] + self.values[:offset])

        def __iter__(self):
            return iter(self.values)

    def load_dataset(dataset, config, **kwargs):
        assert (dataset, config) == ("unit/stream", "unit-config")
        assert kwargs == {"split": "train", "streaming": True, "revision": "unit-revision"}
        return Stream(list(rows))

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    args = SimpleNamespace(
        dataset="unit/stream",
        dataset_config="unit-config",
        dataset_revision="unit-revision",
        text_field="payload.body",
        train_split="train",
        validation_split="",
        streaming=True,
        streaming_shuffle_buffer=10,
        streaming_max_source_tokens=100,
        max_documents=8,
        max_validation_documents=8,
        validation_fraction=0.2,
        seed=3,
    )
    train_a, validation_a = load_document_splits(args)
    args.seed = 19
    train_b, validation_b = load_document_splits(args)

    assert len(train_a) == len(train_b) == len(validation_a) == len(validation_b) == 8
    assert validation_a == validation_b
    assert not (set(train_a) & set(validation_a))
    assert not (set(train_b) & set(validation_b))
    assert "never keep me" not in {*train_a, *validation_a}


def test_pretrain_gate_requires_quality_and_repetition_probe_rejects_collapse(tmp_path):
    path = tmp_path / "unready.pt"
    torch.save({"pretrain_quality": {"passed": False}}, path)
    with pytest.raises(ValueError, match="quality gate"):
        require_pretrain_ready(path)
    assert _healthy_completion([1, 2, 3, 4, 5, 6, 7, 8], eos_token_id=9)
    assert not _healthy_completion([1, 1, 1, 1], eos_token_id=9)
    assert not _healthy_completion([1, 2, 3, 4, 1, 2, 3, 5], eos_token_id=9)
    # One repeated phrase in a long natural continuation is not a loop.
    assert _healthy_completion(list(range(1, 20)) + [1, 2, 3, 20], eos_token_id=99)


class _GroundedTokenizer:
    eos_token_id = 0

    def __init__(self):
        self._ids = {"<eos>": self.eos_token_id}
        self._words = {self.eos_token_id: "<eos>"}

    def encode(self, text, **_kwargs):
        ids = []
        for word in text.split():
            if word not in self._ids:
                token_id = len(self._ids)
                self._ids[word] = token_id
                self._words[token_id] = word
            ids.append(self._ids[word])
        return ids

    def decode(self, token_ids, **_kwargs):
        return " ".join(self._words[token_id] for token_id in token_ids if token_id != self.eos_token_id)


class _GroundedModel:
    config = SimpleNamespace(max_seq_len=64)

    def __init__(self, outputs):
        self.outputs = outputs
        self.generate_kwargs = []

    def generate(self, input_ids, **_kwargs):
        self.generate_kwargs.append(_kwargs)
        return torch.tensor([[999, *self.outputs[int(input_ids[0, 0])]]], dtype=torch.long)


def _grounded_examples(tokenizer):
    examples = []
    for index in range(8):
        prompt = f"prompt{index} context introduction background question passage sentence item topic"
        target = " ".join(f"gold{index}term{part}" for part in range(16))
        examples.append(
            {
                "prompt": prompt,
                "target": target,
                "prompt_token_ids": tokenizer.encode(prompt),
                "target_token_ids": tokenizer.encode(target),
            }
        )
    return examples


def test_long_rollout_reports_repeated_ngram_rate_without_changing_panel_identity():
    tokenizer = _GroundedTokenizer()
    examples = _grounded_examples(tokenizer)
    repeated = tokenizer.encode("loop one two three loop one two three loop")
    outputs = {example["prompt_token_ids"][0]: repeated for example in examples}
    model = _GroundedModel(outputs)
    spec = _grounded_rollout_spec(examples, tokenizer)

    result = evaluate_grounded_rollouts(model, tokenizer, examples, "cpu", spec=spec, generated_tokens=17)

    assert spec["case_indices"] == list(range(8))
    assert result["case_count"] == 8
    assert result["healthy_fraction"] == 0.0
    assert result["generated_tokens"] == 17
    assert all(kwargs["max_new_tokens"] == 17 for kwargs in model.generate_kwargs)
    assert result["mean_repeated_4gram_fraction"] > 0
    assert spec["generated_tokens"] == 16
    assert _repeated_ngram_fraction([2, 3, 4, 5, 2, 3, 4, 5], eos_token_id=0) == pytest.approx(0.2)


def test_checkpoint_metadata_keeps_serialized_foundation_evaluation_examples(tmp_path):
    path = tmp_path / "foundation.pt"
    foundation_examples = [{"prompt": "A prompt", "target": "A continuation", "target_eos": True}]
    torch.save(
        {
            "pretrain_quality": {
                "gate_version": PRETRAIN_QUALITY_GATE_VERSION,
                "passed": True,
                "healthy_probes_ready": True,
                "foundation_healthy_probes_ready": True,
                "conditioning_ready": True,
                "foundation_conditioning_ready": True,
                "foundation_evaluation_examples": foundation_examples,
            }
        },
        path,
    )
    require_pretrain_ready(path)
    assert checkpoint_metadata(path)["pretrain_quality"]["foundation_evaluation_examples"] == foundation_examples


def test_evolution_reward_prefers_reference_matches_without_penalizing_short_answers():
    assert reward_for_answer("Neutral", "Neutral", ended=True) == pytest.approx(1.05)
    assert reward_for_answer("Final answer: -5", "-5", ended=True) == pytest.approx(1.05)
    assert reward_for_answer("Final answer: 5", "-5", ended=True) < 1
    assert reward_for_answer("Final answer: A or B", "A", ended=False) == 0
    assert reward_for_answer("100 100 100 100", "42", ended=False) < 0
