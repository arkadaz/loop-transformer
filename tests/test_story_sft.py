from src.data import story_checks, tinystories_instruct
from src.evolution import reward_for_story
from src.gemma_teacher import TeacherConfig, _clean_story, augment_with_gemma

ROW = {
    "Features": "Dialogue, MoralValue",
    "Words": "ride, work, upset",
    "Summary": "Lily learns to share her bike.",
    "story": " ".join(["Lily liked to ride her bike after work, but she got upset when Ben asked to share."] * 6),
}


def test_instruct_formatter_builds_prompt_and_verifier_fields():
    example = tinystories_instruct(ROW, "tinystories_instruct")
    assert example["prompt"].startswith("Write a short story for young children. Use the words: ride, work, upset.")
    assert "Include dialogue." in example["prompt"]
    assert example["required_words"] == ["ride", "work", "upset"] and example["needs_dialogue"]
    assert tinystories_instruct({**ROW, "story": "too short"}, "x")["target"] == ""


def test_story_checks_and_reward():
    example = tinystories_instruct(ROW, "tinystories_instruct")
    assert story_checks('Lily said, "I ride to work." She was upset.', example) == (1.0, True)
    assert story_checks("Nothing relevant here", example) == (0.0, False)
    good = 'Lily said, "I ride to work every day." ' + " ".join(f"word{i}" for i in range(30)) + " She was upset."
    assert reward_for_story(good, example, ended=True) > 1.1
    assert reward_for_story("too short", example, ended=True) == 0.0
    looped = "ride work upset " * 30
    assert reward_for_story(looped, example, ended=False) < 0.5


def test_gemma_story_mode_uses_verified_story_or_falls_back(tmp_path):
    example = tinystories_instruct(ROW, "tinystories_instruct")
    story = 'Here is a story:\nLily loved to ride her bike to work. "Can I try?" asked Ben. Lily was upset, but she shared. ' + "They laughed together all afternoon. " * 5
    accepted = augment_with_gemma([example], cache_path=tmp_path / "a.jsonl", response_generator=lambda p, a: (story, True))
    assert accepted[0]["teacher_accepted"] and accepted[0]["target"].startswith("Lily loved to ride")
    fallback = augment_with_gemma([example], cache_path=tmp_path / "b.jsonl", response_generator=lambda p, a: ("A cat sat. " * 30, True))
    assert not fallback[0]["teacher_accepted"] and fallback[0]["target"] == ROW["story"]
    assert _clean_story("**Title**\nHere is the story:\nOnce there was a small dog who loved bones.") == "Once there was a small dog who loved bones."


def test_continuation_replay_skips_the_teacher(tmp_path):
    from src.data import tinystories_continue

    replay = tinystories_continue(ROW)
    assert replay["mode"] == "anchor" and replay["prompt"] and replay["target"]
    assert tinystories_continue({"story": "One. Two."})["target"] == ""
    out = augment_with_gemma([replay], cache_path=tmp_path / "c.jsonl", response_generator=lambda p, a: ("unused", True))
    assert out == [replay] and not (tmp_path / "c.jsonl").exists()


def test_name_cap_limits_repeated_character_names(monkeypatch):
    import io

    import src.data as data

    class FakeFS:
        def open(self, *_args, **_kwargs):
            names = ["Lily"] * 8 + ["Ben", "Mia", "Sam", "Tom"]
            body = "".join(f"Words: a, b\nStory: \n\n{n} was a girl. {n} played all day. Summary: trailing field\n<|endoftext|>" for n in names)
            return io.BytesIO(("<|endoftext|>" + body + "tail").encode())

    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda: FakeFS())
    rows = data._instruct_rows("x.txt", 10, 1, name_cap=0.2)
    assert sum("Lily" in r["story"] for r in rows) == 2 and len(rows) == 10
    assert all("Lily" not in r["story"] or r["story"].count("Lily") == 2 for r in rows)  # renamed rows lose every mention
    assert len(data._instruct_rows("x.txt", 10, 1)) == 10


def test_parser_keeps_trailing_fields_out_of_the_story(monkeypatch):
    import io

    import src.data as data

    class FakeFS:
        def open(self, *_args, **_kwargs):
            entry = "Story: \n\nMia had a kite.\nIt flew high.\nSummary: Mia flies a kite.\nWords: kite, high, fun\n<|endoftext|>"
            return io.BytesIO(("<|endoftext|>" + entry + "tail").encode())

    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda: FakeFS())
    (row,) = data._instruct_rows("x.txt", 5, 1)
    assert row["story"] == "Mia had a kite.\nIt flew high." and row["Words"] == "kite, high, fun" and row["Summary"]
    assert data._protagonist("Once upon a time, Lily saw Tom. Lily laughed. Mom smiled.") == "Lily"
