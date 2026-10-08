"""RQ4: what characteristics distinguish the LLM families?

Usage: python src/analyze_features.py

Writes to results/rq4/:
  feature_means.csv      per-family mean/median of each hand-made feature
  feature_groups.json    accuracy of a classifier on feature groups (and with one group removed)
  top_tokens.csv         tokens most associated with each family (training split only)
  opening_words.csv      most common first word of each family's responses
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "llm_dataset.csv"
OUT = ROOT / "results" / "rq4"

TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def per100(count, n_tokens):
    return 100.0 * count / max(n_tokens, 1)


def features(text):
    toks = TOKEN_RE.findall(text.lower())
    words = [t for t in toks if re.match(r"\w", t)]
    n_tok, n_w = len(toks), max(len(words), 1)
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    first100 = words[:100]
    f = {
        # --- length ---
        "n_chars": len(text),
        "n_words": len(words),
        "n_lines": len(nonblank),
        "n_sentences": len(sents),
        "avg_sentence_words": len(words) / max(len(sents), 1),
        # --- lexical ---
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "ttr_first100": len(set(first100)) / max(len(first100), 1),
        "hapax_ratio": sum(1 for c in Counter(words).values() if c == 1) / n_w,
        "i_rate": per100(sum(w in ("i", "i'm", "my") for w in words), n_w),
        "you_rate": per100(sum(w in ("you", "your") for w in words), n_w),
        "apology": int(bool(re.search(r"\b(sorry|apologi[sz]e)\b", text.lower()))),
        "as_an_ai": int("as an ai" in text.lower() or "language model" in text.lower()),
        "cannot": int(bool(re.search(r"\b(cannot|can't|unable to)\b", text.lower()))),
        # --- punctuation ---
        "comma_rate": per100(text.count(","), n_tok),
        "period_rate": per100(text.count("."), n_tok),
        "excl_rate": per100(text.count("!"), n_tok),
        "question_rate": per100(text.count("?"), n_tok),
        "colon_rate": per100(text.count(":"), n_tok),
        "quote_rate": per100(text.count('"'), n_tok),
        "paren_rate": per100(text.count("("), n_tok),
        # --- structure / formatting ---
        "n_newlines": text.count("\n"),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        "avg_line_words": len(words) / max(len(nonblank), 1),
        "bullet_lines": sum(bool(re.match(r"\s*[-*•]\s", ln)) for ln in lines),
        "numbered_lines": sum(bool(re.match(r"\s*\d+[.)]\s", ln)) for ln in lines),
        "bold_markers": text.count("**"),
        "header_lines": sum(bool(re.match(r"\s*#{1,6}\s", ln)) for ln in lines),
        "code_blocks": text.count("```") // 2,
        "starts_with_here": int(bool(re.match(r"\W*(sure|certainly|here)", text.lower()))),
        "ends_with_question": int(text.rstrip().endswith("?")),
    }
    return f


GROUPS = {
    "length": ["n_chars", "n_words", "n_lines", "n_sentences", "avg_sentence_words"],
    "lexical": [
        "avg_word_len", "ttr_first100", "hapax_ratio", "i_rate", "you_rate",
        "apology", "as_an_ai", "cannot",
    ],
    "punctuation": [
        "comma_rate", "period_rate", "excl_rate", "question_rate",
        "colon_rate", "quote_rate", "paren_rate",
    ],
    "formatting": [
        "n_newlines", "n_paragraphs", "avg_line_words", "bullet_lines",
        "numbered_lines", "bold_markers", "header_lines", "code_blocks",
        "starts_with_here", "ends_with_question",
    ],
}


def fit_eval(X_tr, y_tr, X_te, y_te, kind):
    if kind == "logreg":
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X_tr), y_tr)
        pred = clf.predict(sc.transform(X_te))
    else:
        clf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        pred = clf.predict(X_te)
    return accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro"), clf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    feats = pd.DataFrame([features(t) for t in df.llm_output])
    feats["llm_name"] = df.llm_name.values
    feats["split"] = df.split.values

    # 1) per-family descriptive statistics
    cols = [c for g in GROUPS.values() for c in g]
    means = feats.groupby("llm_name")[cols].agg(["mean", "median"]).T
    means.to_csv(OUT / "feature_means.csv")
    print("\n=== Mean per family ===")
    print(feats.groupby("llm_name")[cols].mean().T.round(3).to_string())

    # 2) classifiers on hand-made feature groups
    tr, te = feats[feats.split == "train"], feats[feats.split == "test"]
    y_tr, y_te = tr.llm_name, te.llm_name
    results = {}

    def run(name, cs):
        for kind in ("logreg", "forest"):
            acc, f1, _ = fit_eval(tr[cs], y_tr, te[cs], y_te, kind)
            results.setdefault(name, {})[kind] = {"acc": acc, "macro_f1": f1}
        print(f"{name:28s} logreg {results[name]['logreg']['acc']:.3f} | forest {results[name]['forest']['acc']:.3f}")

    print("\n=== Test accuracy, feature groups (chance = 0.333) ===")
    run("all_features", cols)
    for g, cs in GROUPS.items():
        run(f"only_{g}", cs)
    for g, drop in GROUPS.items():
        run(f"all_minus_{g}", [c for c in cols if c not in drop])
    # structural vs linguistic as the assignment words it
    run("structural(length+format+punct)", GROUPS["length"] + GROUPS["formatting"] + GROUPS["punctuation"])
    run("linguistic(lexical)", GROUPS["lexical"])
    (OUT / "feature_groups.json").write_text(json.dumps(results, indent=2))

    # random forest feature importances (all features)
    _, _, clf = fit_eval(tr[cols], y_tr, te[cols], y_te, "forest")
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
    print("\n=== Top 10 random-forest feature importances ===")
    print(imp.head(10).round(3).to_string())

    # 3) most family-specific tokens (smoothed log-odds, training split only)
    train_df = df[df.split == "train"]
    counts = {
        fam: Counter(t for text in g.llm_output for t in set(TOKEN_RE.findall(text.lower())))
        for fam, g in train_df.groupby("llm_name")
    }
    docs = train_df.llm_name.value_counts().to_dict()
    vocab = {t for c in counts.values() for t, n in c.items() if n >= 20}
    rows = []
    for fam in counts:
        for t in vocab:
            a = counts[fam][t] + 1
            b = sum(counts[o][t] for o in counts if o != fam) + 1
            na = docs[fam] + 2
            nb = sum(docs[o] for o in counts if o != fam) + 2
            rows.append((fam, t if t != "\n" else "<newline>", np.log(a / na) - np.log(b / nb), counts[fam][t]))
    tok = pd.DataFrame(rows, columns=["family", "token", "log_odds", "docs_with_token"])
    top = tok.sort_values("log_odds", ascending=False).groupby("family").head(25)
    top.to_csv(OUT / "top_tokens.csv", index=False)
    print("\n=== Most family-specific tokens (log-odds) ===")
    for fam, g in top.groupby("family"):
        print(fam, ":", ", ".join(g.token.tolist()[:20]))

    # 4) opening words
    first = df.llm_output.map(lambda t: (TOKEN_RE.findall(t.lower()) or [""])[0])
    op = pd.crosstab(first, df.llm_name)
    op["total"] = op.sum(axis=1)
    op = op.sort_values("total", ascending=False).head(20)
    op.to_csv(OUT / "opening_words.csv")
    print("\n=== Most common opening words (counts) ===")
    print(op.to_string())


if __name__ == "__main__":
    main()
