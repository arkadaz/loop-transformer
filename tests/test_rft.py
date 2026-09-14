import json

from src.data import load_examples
from src.rft import FULL_PASS, best_candidate

EXAMPLE = {"required_words": ["ride", "work", "upset"], "needs_dialogue": True}
GOOD = 'Lily said, "I ride to work every day." ' + " ".join(f"word{i}" for i in range(30)) + " She was upset but happy."
PARTIAL = 'Lily liked to ride. ' + " ".join(f"other{i}" for i in range(30)) + " The end."


def test_best_candidate_requires_a_full_verifier_pass():
    index, reward = best_candidate([PARTIAL, GOOD], [True, True], EXAMPLE)
    assert index == 1 and reward >= FULL_PASS
    index, reward = best_candidate([PARTIAL, "short"], [True, True], EXAMPLE)
    assert index is None and reward < FULL_PASS
    assert best_candidate([PARTIAL, "short"], [True, True], EXAMPLE, min_reward=-1.0)[0] == 0


def test_jsonl_source_loads_local_examples(tmp_path):
    path = tmp_path / "rft.jsonl"
    rows = [{"prompt": f"p{i}", "target": GOOD, "source": "rft", "mode": "anchor"} for i in range(5)] + [{"prompt": "empty", "target": ""}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    examples = load_examples((f"jsonl:{path}",), per_source=3, seed=1)
    assert len(examples) == 3 and all(e["mode"] == "anchor" and e["target"] for e in examples)
