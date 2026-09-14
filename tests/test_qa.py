from src.data import QA_PROMPT, qa_acceptable, qa_checks, tinystories_qa
from src.evolution import reward_for_qa
from src.gemma_teacher import augment_with_gemma

STORY = (
    "Mia found a red kite in the garden. She ran with it under the sun. "
    "The kite flew high above the trees. Her brother Sam clapped his hands. "
    "They laughed until the sky turned orange and it was time for dinner."
)
PASSAGE = tinystories_qa({"story": STORY}, "tinystories_qa")


def test_passage_takes_whole_sentences_and_rejects_short_stories():
    assert PASSAGE["mode"] == "qa" and PASSAGE["target"] == "" and PASSAGE["passage"].startswith("Mia found a red kite")
    assert PASSAGE["passage"].endswith(".") and len(PASSAGE["passage"].split()) <= 55
    assert tinystories_qa({"story": "Too short."}, "x").get("mode") != "qa"


def test_acceptance_requires_a_short_grounded_answer():
    assert qa_acceptable("What did Mia find in the garden?", "A red kite.", PASSAGE["passage"])
    assert not qa_acceptable("What did Mia find in the garden?", "A blue submarine from Denmark.", PASSAGE["passage"])
    assert not qa_acceptable("No question mark here", "A red kite.", PASSAGE["passage"])
    assert not qa_acceptable("What did Mia find?", " ".join(["kite"] * 20), PASSAGE["passage"])


def test_f1_grounding_and_reward():
    example = {"reference_answer": "a red kite", "passage": PASSAGE["passage"]}
    assert qa_checks("a red kite", example) == (1.0, 1.0)
    partial_f1, grounded = qa_checks("a red balloon", example)
    assert 0 < partial_f1 < 1 and grounded < 1
    assert qa_checks("", example) == (0.0, 0.0)
    assert reward_for_qa("a red kite", example, ended=True) == 1.05
    assert reward_for_qa("a red kite", example, ended=False) == 1.0


def test_gemma_qa_mode_builds_the_prompt_and_drops_unverified_pairs(tmp_path):
    good = "Question: What did Mia find in the garden?\nAnswer: A red kite."
    (example,) = augment_with_gemma([PASSAGE], cache_path=tmp_path / "a.jsonl", response_generator=lambda p, a: (good, True))
    assert example["prompt"] == QA_PROMPT.format(passage=PASSAGE["passage"], question="What did Mia find in the garden?")
    assert example["target"] == example["reference_answer"] == "A red kite." and example["mode"] == "anchor"

    bad = "Question: What did Mia find?\nAnswer: A submarine from Denmark."
    assert augment_with_gemma([PASSAGE], cache_path=tmp_path / "b.jsonl", response_generator=lambda p, a: (bad, True)) == []
    assert augment_with_gemma([PASSAGE], cache_path=tmp_path / "c.jsonl", response_generator=lambda p, a: ("no format", True)) == []
