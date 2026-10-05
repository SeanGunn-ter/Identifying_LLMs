"""Build an (llm_name, llm_input, llm_output) dataset from LMSYS Arena 55k."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train.csv"
OUT = ROOT / "data" / "processed" / "llm_dataset.csv"

FAMILY_PREFIXES = {
    "gpt-": "GPT",
    "claude": "Claude",
    "gemini": "Gemini",
}
MAX_PER_CLASS = 2500
MIN_PER_CLASS = 1500


def family(model):
    m = model.lower()
    for prefix, fam in FAMILY_PREFIXES.items():
        if m.startswith(prefix):
            return fam
    return None  # vicuna, etc. are excluded


def first_turn(field):
    try:
        turns = json.loads(field)
        text = turns[0] if turns else None
        return text.strip() if isinstance(text, str) and text.strip() else None
    except (json.JSONDecodeError, TypeError):
        return None


def task_type(prompt):
    """Rough task label, used for RQ3 cross-domain experiments."""
    p = prompt.lower()
    if "```" in prompt or re.search(
        r"\b(python|java|javascript|c\+\+|sql|code|function|script|bug)\b", p
    ):
        return "coding"
    if re.search(
        r"\b(solve|calculate|equation|math|integral|probability)\b", p
    ) or re.search(r"\d+\s*[-+*/^]\s*\d+", p):
        return "math"
    if re.search(r"\b(write|story|poem|essay|email|letter|lyrics|rewrite)\b", p):
        return "writing"
    return "qa"


def split_for(prompt_id):
    """Split by prompt so the same prompt never appears in both train and test."""
    h = int(hashlib.md5(str(prompt_id).encode()).hexdigest(), 16) % 10
    return "train" if h < 7 else ("val" if h < 8 else "test")


def main():
    df = pd.read_csv(RAW)
    rows = []
    for r in df.itertuples():
        prompt = first_turn(r.prompt)
        if prompt is None:
            continue
        for model, resp in ((r.model_a, r.response_a), (r.model_b, r.response_b)):
            fam, out = family(model), first_turn(resp)
            if fam and out:
                rows.append(
                    {
                        "prompt_id": r.id,
                        "llm_name": fam,
                        "model_version": model,
                        "llm_input": prompt,
                        "llm_output": out,
                        "task_type": task_type(prompt),
                        "split": split_for(r.id),
                    }
                )

    data = pd.DataFrame(rows)
    print("Counts before balancing:\n", data.llm_name.value_counts(), "\n")

    counts = data.llm_name.value_counts()
    keep = counts[counts >= MIN_PER_CLASS].index
    n = min(MAX_PER_CLASS, counts[keep].min())
    data = (
        data[data.llm_name.isin(keep)].groupby("llm_name").sample(n=n, random_state=42)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data)} rows ({n} per class, families: {list(keep)}) to {OUT}")
    print(pd.crosstab(data.llm_name, data.task_type))
    print(data.split.value_counts())


if __name__ == "__main__":
    main()
