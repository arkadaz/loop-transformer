"""Small, answer-anchored datasets for Gemma-assisted student training."""
from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


Example = dict[str, Any]


def _compact(text: Any, limit: int = 1_600) -> str:
    return " ".join(str(text).replace("\n", " ").split())[:limit]


def _choices(choices: Mapping[str, Any], answer_key: str) -> tuple[str, str]:
    labels = list(choices["label"])
    texts = list(choices["text"])
    index = labels.index(answer_key)
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return rendered, str(texts[index])


def gsm8k(row: Mapping[str, Any]) -> Example:
    answer = str(row["answer"])
    final = answer.rsplit("####", 1)[-1].strip()
    return {
        "prompt": f"Solve the math problem. Give the final answer.\n\n{_compact(row['question'])}",
        "target": final,
        "source": "gsm8k",
    }


def arc(row: Mapping[str, Any], source: str) -> Example:
    choices, answer = _choices(row["choices"], str(row["answerKey"]))
    return {
        "prompt": f"Choose the best science answer.\n\n{_compact(row['question'])}\n\n{choices}",
        "target": answer,
        "source": source,
    }


def commonsense_qa(row: Mapping[str, Any]) -> Example:
    choices, answer = _choices(row["choices"], str(row["answerKey"]))
    return {
        "prompt": f"Choose the most sensible answer.\n\n{_compact(row['question'])}\n\n{choices}",
        "target": answer,
        "source": "commonsenseqa",
    }


def finance_sentiment(row: Mapping[str, Any]) -> Example:
    labels = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    return {
        "prompt": f"Classify the market sentiment as Bearish, Bullish, or Neutral.\n\n{_compact(row['text'])}",
        "target": labels[int(row["label"])],
        "source": "finance_sentiment",
    }


def dolly(row: Mapping[str, Any]) -> Example:
    context = _compact(row.get("context", ""), 800)
    prompt = _compact(row["instruction"], 800)
    if context:
        prompt = f"{prompt}\n\nContext: {context}"
    response = " ".join(str(row["response"]).split())
    return {"prompt": prompt, "target": response[:800], "target_eos": len(response) <= 800, "source": "dolly"}


_INSTRUCT_REPO = "datasets/roneneldan/TinyStoriesInstruct"
_QUOTES = ('"', "\u201c", "\u201d")
_FIELDS = frozenset({"Summary", "Words", "Features", "Random sentence"})
_CAPITALISED = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NOT_NAMES = frozenset(
    "Once One The There She He It They We You His Her Then But And So When After Suddenly This That What Why How Can Let "
    "Look Yes Thank Oh Wow Hi Hello Bye Every From Now Just Come Please Okay Good Little Big Mom Mum Dad Mommy Mummy Daddy "
    "Grandma Grandpa Mama Papa Mother Father Day Was Were Did Not With For Are Have Had Don Didn Its Him Them Their All Some "
    "Many Two Three Everyone Summary Words Features Story".split()
)


_NAME_POOL = (
    "Ava Ben Bella Cara Dan Eli Emma Finn Gus Hana Ivy Jay Kai Leo Lena Milo Nina Omar Pia Quinn Rosa Sam Tara Uma Vic "
    "Wren Zoe Ali Amir Chen Dev Ines Jun Kofi Lila Mateo Noor Ola Ravi Sofia Tomas Yara Ezra Freya Hugo Isla Nico Ruby"
).split()


def _protagonist(story: str) -> str:
    """Most frequent capitalised word that is not a common sentence starter or family noun."""
    counts = Counter(word for word in _CAPITALISED.findall(story) if word not in _NOT_NAMES)
    return counts.most_common(1)[0][0] if counts else ""


def _instruct_rows(filename: str, per_source: int, seed: int, name_cap: float | None = None) -> list[dict[str, str]]:
    """Read the head of one TinyStoriesInstruct text file; entries are separated by <|endoftext|>.

    ``name_cap`` limits the share of rows sharing one protagonist name by renaming the
    protagonist in over-represented rows: TinyStories over-uses a few names and a small
    student collapses onto them, so the head of the name distribution is flattened.
    """
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"{_INSTRUCT_REPO}/{filename}", "rb") as handle:
        text = handle.read(max(12_000_000, per_source * 2_500)).decode("utf-8", errors="ignore")  # ~1.1 KB per entry
    rows = []
    for entry in text.split("<|endoftext|>")[1:-1]:
        fields, story, in_story = {}, [], False
        for line in entry.strip().splitlines():
            key, sep, value = line.partition(":")
            if key.strip() == "Story":
                in_story = True
            elif key.strip() in _FIELDS:
                fields[key.strip()], in_story = value.strip(), False
            elif in_story:
                story.append(line.strip())
        fields["story"] = "\n".join(line for line in story if line)
        rows.append(fields)
    random.Random(seed).shuffle(rows)
    if name_cap is None:
        return rows[:per_source]
    rows, names, limit, rng = rows[:per_source], Counter(), max(1, int(name_cap * per_source)), random.Random(seed)
    for row in rows:
        name = _protagonist(row["story"])
        if name and names[name] >= limit:
            new_name = rng.choice([n for n in _NAME_POOL if n != name])
            row.update({key: re.sub(rf"\b{name}\b", new_name, value) for key, value in row.items()})
            name = new_name
        names[name] += 1
    return rows


def tinystories_instruct(row: Mapping[str, Any], source: str) -> Example:
    words = [word.strip().lower() for word in row.get("Words", "").split(",") if word.strip()]
    story = row.get("story", "")
    if not words or len(story.split()) < 40:
        return {"prompt": "", "target": "", "source": source}
    dialogue = "dialogue" in row.get("Features", "").lower()
    summary = _compact(row.get("Summary", ""), 300)
    prompt = f"Write a short story for young children. Use the words: {', '.join(words)}."
    if dialogue:
        prompt += " Include dialogue."
    if summary and len(story) % 2:  # half the prompts carry a plot so both styles are learned
        prompt += f" The story is about: {summary}"
    sentence = _compact(row.get("Random sentence", ""), 200)
    if sentence:  # a required sentence forces different openings and plots
        prompt += f" Include this sentence: {sentence}"
    return {"prompt": prompt, "target": story, "source": source, "mode": "story", "required_words": words, "needs_dialogue": dialogue}


def tinystories_continue(row: Mapping[str, Any]) -> Example:
    """Teacher-free replay: a story prefix continues into the rest of the same story."""
    sentences = re.split(r"(?<=[.!?])\s+", row.get("story", "").replace("\n", " ").strip())
    if len(sentences) < 4 or len(" ".join(sentences).split()) < 40:
        return {"prompt": "", "target": "", "source": "tinystories_continue"}
    cut = max(1, len(sentences) // 3)
    # The leading space matches how pretraining continuations are tokenized.
    return {"prompt": " ".join(sentences[:cut]), "target": " " + " ".join(sentences[cut:]), "source": "tinystories_continue", "mode": "anchor"}


def story_checks(text: str, example: Mapping[str, Any]) -> tuple[float, bool]:
    """Return (share of required words present, dialogue requirement satisfied)."""
    lowered = text.lower()
    words = example.get("required_words") or []
    hits = sum(bool(re.search(rf"\b{re.escape(word)}", lowered)) for word in words)
    dialogue_ok = not example.get("needs_dialogue") or any(quote in text for quote in _QUOTES)
    return (hits / len(words) if words else 1.0), dialogue_ok


@dataclass(frozen=True)
class DatasetSource:
    repository: str
    config: str | None
    formatter: Callable[[Mapping[str, Any]], Example]
    license_note: str
    rows: Callable[[int, int], list[dict[str, Any]]] | None = None  # custom loader: (per_source, seed) -> rows


SOURCES: dict[str, DatasetSource] = {
    "gsm8k": DatasetSource("openai/gsm8k", "main", gsm8k, "MIT"),
    "arc_easy": DatasetSource("allenai/ai2_arc", "ARC-Easy", lambda row: arc(row, "arc_easy"), "CC-BY-SA-4.0"),
    "arc_challenge": DatasetSource("allenai/ai2_arc", "ARC-Challenge", lambda row: arc(row, "arc_challenge"), "CC-BY-SA-4.0"),
    "commonsenseqa": DatasetSource("tau/commonsense_qa", None, commonsense_qa, "MIT"),
    "finance_sentiment": DatasetSource(
        "zeroshot/twitter-financial-news-sentiment", None, finance_sentiment, "MIT"
    ),
    "dolly": DatasetSource("databricks/databricks-dolly-15k", None, dolly, "CC-BY-SA-3.0"),
    "tinystories_instruct": DatasetSource(
        "roneneldan/TinyStoriesInstruct", None, lambda row: tinystories_instruct(row, "tinystories_instruct"),
        "CDLA-Sharing-1.0", rows=lambda n, seed: _instruct_rows("TinyStories-Instruct-train.txt", n, seed, name_cap=0.05),
    ),
    "tinystories_continue": DatasetSource(
        "roneneldan/TinyStoriesInstruct", None, tinystories_continue,
        "CDLA-Sharing-1.0", rows=lambda n, seed: _instruct_rows("TinyStories-Instruct-train.txt", n, seed, name_cap=0.05),
    ),
    "tinystories_instruct_valid": DatasetSource(
        "roneneldan/TinyStoriesInstruct", None, lambda row: tinystories_instruct(row, "tinystories_instruct_valid"),
        "CDLA-Sharing-1.0", rows=lambda n, seed: _instruct_rows("TinyStories-Instruct-valid.txt", n, seed),
    ),
}

# Dolly supplies general single-turn instruction following. The remaining
# sources have trusted short answers, which is important for later verifiers.
DEFAULT_SOURCES = ("gsm8k", "arc_easy", "arc_challenge", "commonsenseqa", "finance_sentiment", "dolly")


def load_examples(
    source_names: tuple[str, ...] = DEFAULT_SOURCES,
    *,
    per_source: int = 300,
    seed: int = 42,
) -> list[Example]:
    """Load a deterministic, balanced sample from each selected train split."""
    if per_source < 1:
        raise ValueError("per_source must be positive.")
    from datasets import load_dataset

    examples: list[Example] = []
    failures: list[str] = []
    for offset, name in enumerate(source_names):
        if name not in SOURCES:
            raise ValueError(f"Unknown dataset source {name!r}. Choose from {', '.join(SOURCES)}.")
        source = SOURCES[name]
        try:
            print(f"Loading {name}...")
            if source.rows is not None:
                sample = source.rows(per_source, seed + offset)
            else:
                dataset = load_dataset(source.repository, source.config, split="train")
                sample = dataset.shuffle(seed=seed + offset).select(range(min(per_source, len(dataset))))
            loaded = 0
            for row in sample:
                example = source.formatter(row)
                if example["target"]:
                    examples.append(example)
                    loaded += 1
            print(f"Loaded {loaded} {name} examples.")
        except Exception as error:  # Network, gated data, and legacy builders vary by environment.
            failures.append(f"{name}: {error}")

    if not examples:
        detail = "\n".join(failures) or "No rows were returned."
        raise RuntimeError(f"Could not load any training examples.\n{detail}")
    if failures:
        print("Dataset warnings:\n" + "\n".join(failures))
    return examples
