from types import SimpleNamespace

import pytest
import torch

from src.data import dolly
from src.gemma_teacher import TeacherConfig, augment_with_gemma, generate_explanations
from src.pretrain_distill import TokenPairs, tokenize_target


class CharacterTokenizer:
    eos_token_id = 999

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


@pytest.mark.parametrize("text,ended,expected", [
    ("ab", True, [97, 98, 999]),
    ("abc", True, [97, 98, 99]),
    ("abcd", True, [97, 98, 99]),
    ("ab", False, [97, 98]),
])
def test_student_eos_requires_natural_end_and_room(text, ended, expected):
    tokenizer = CharacterTokenizer()
    assert tokenize_target(tokenizer, text, 3, append_eos=ended) == expected
    for exact_ids in (False, True):
        example = {"prompt": "p", "target": text, "target_eos": ended}
        if exact_ids:
            example["target_token_ids"] = tokenizer.encode(text)
        assert TokenPairs([example], tokenizer, 8, 3)[0][1] == expected


@pytest.mark.parametrize("ended", [False, True])
def test_teacher_completion_status_survives_cache_and_student_encoding(tmp_path, ended):
    args = {
        "examples": [{"prompt": "Question", "target": "Answer", "source": "unit"}],
        "cache_path": tmp_path / "targets.jsonl",
    }
    augmented = augment_with_gemma(
        **args, response_generator=lambda *_: ("A supported explanation.", ended),
    )
    cached = augment_with_gemma(
        **args, response_generator=lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert augmented == cached
    assert augmented[0]["target_eos"] is ended
    target = TokenPairs(augmented, CharacterTokenizer(), 256, 256)[0][1]
    assert (target[-1] == 999) is ended


def test_unknown_or_capped_explanation_does_not_remove_complete_answer_eos_when_dropped(tmp_path):
    result = augment_with_gemma(
        [{"prompt": "Question", "target": "Answer", "source": "unit"}],
        cache_path=tmp_path / "targets.jsonl", response_generator=lambda *_: ("", False),
    )
    assert result[0]["target"] == "Final answer: Answer"
    assert result[0]["target_eos"] is True


def test_teacher_turn_stop_is_complete_but_token_cap_is_not():
    class Tokenizer:
        pad_token_id, eos_token_id, padding_side = 0, 1, "right"

        def apply_chat_template(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[4, 5], [4, 5]]), "attention_mask": torch.ones(2, 2)}

        def decode(self, tokens, **_kwargs):
            return " ".join(str(token) for token in tokens.tolist() if token not in {0, 1, 106})

    class Teacher:
        generation_config = SimpleNamespace(eos_token_id=[1, 106])

        def generate(self, **kwargs):
            assert kwargs["eos_token_id"] == [1, 106]
            return torch.tensor([[4, 5, 7, 106], [4, 5, 8, 9]])

    assert generate_explanations(
        Teacher(), Tokenizer(), [("a", "b"), ("c", "d")], TeacherConfig(max_new_tokens=2), "cpu",
        return_completion_flags=True,
    ) == [("7", True), ("8 9", False)]


def test_dolly_character_truncation_marks_target_open():
    assert dolly({"instruction": "Explain", "response": "a" * 801})["target_eos"] is False
    assert dolly({"instruction": "Explain", "response": "Finished."})["target_eos"] is True
