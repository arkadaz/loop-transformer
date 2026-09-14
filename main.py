import sys

from src import evolution, pretrain, pretrain_distill


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"pretrain", "anneal", "posttrain", "sft", "evolve"}:
        raise SystemExit("Usage: python main.py {pretrain|anneal|posttrain|sft|evolve} [options]")
    stage = sys.argv.pop(1)
    {
        "pretrain": pretrain.main,
        "anneal": lambda: pretrain.main(stage="anneal"),
        "posttrain": pretrain_distill.main,
        "sft": pretrain_distill.main,
        "evolve": evolution.main,
    }[stage]()
