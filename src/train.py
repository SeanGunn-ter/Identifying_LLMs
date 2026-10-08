"""Train a CNN or LSTM to identify the LLM family.

Usage: python src/train.py --model cnn --mode output
  --model: cnn | lstm
  --mode:  output | input | both   (RQ1 = output, RQ2 = all three)
"""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.cnn import TextCNN
from models.rnn import TextLSTM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "llm_dataset.csv"
RESULTS = ROOT / "results"

PAD, UNK = 0, 1
MAX_VOCAB, MIN_FREQ = 20000, 2
MAX_OUTPUT = 256
TOKEN_RE = re.compile(
    r"\n|\w+|[^\w\s]"
)  # newlines and punctuation are tokens: formatting is a signal


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


ABLATIONS = [
    "none",
    "no_newlines",  # drop line breaks (paragraph/list layout)
    "no_punct",  # drop punctuation and line breaks
    "no_markdown",  # drop markdown symbols (* # ` - _ > |) and line breaks
    "shuffle",  # random word order: keeps vocabulary, destroys syntax/phrases
    "function_only",  # keep stopwords/punctuation/newlines, mask content words
    "content_only",  # drop stopwords/punctuation/newlines, keep content words
    "skip_first10",  # drop the opening 10 tokens (e.g. "Sure, here is ...")
    "first32",  # only the first 32 tokens (controls for length)
]
MARKDOWN = set("*#`-_>|~")
STOP = set(ENGLISH_STOP_WORDS)


def ablate_tokens(toks, ablate, seed):
    if ablate == "no_newlines":
        return [t for t in toks if t != "\n"]
    if ablate == "no_punct":
        return [t for t in toks if re.match(r"\w", t)]
    if ablate == "no_markdown":
        return [t for t in toks if t != "\n" and t not in MARKDOWN]
    if ablate == "shuffle":
        toks = list(toks)
        random.Random(seed).shuffle(toks)
        return toks
    if ablate == "function_only":
        return [t if (t in STOP or not re.match(r"\w", t)) else "<unk>" for t in toks]
    if ablate == "content_only":
        return [t for t in toks if re.match(r"\w", t) and t not in STOP]
    if ablate == "skip_first10":
        return toks[10:]
    if ablate == "first32":
        return toks[:32]
    return toks


def get_tokens(row, mode, ablate="none"):
    toks = []
    if mode in ("input", "both"):
        toks += tokenize(row.llm_input)[: 256 if mode == "input" else 64]
    if mode == "both":
        toks.append("<sep>")
    if mode in ("output", "both"):
        out = tokenize(row.llm_output)[:MAX_OUTPUT]
        toks += ablate_tokens(out, ablate, row.Index)
    return toks


class TextDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    if x.size(1) < 5:  # CNN needs length >= largest kernel
        x = F.pad(x, (0, 5 - x.size(1)), value=PAD)
    return x, lengths, torch.tensor(labels)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    preds, golds, total_loss = [], [], 0.0
    with torch.set_grad_enabled(train):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds += logits.argmax(1).cpu().tolist()
            golds += y.cpu().tolist()
    return total_loss / len(golds), preds, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--mode", choices=["output", "input", "both"], default="output")
    ap.add_argument("--ablate", choices=ABLATIONS, default="none")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA)
    df["tokens"] = [get_tokens(r, args.mode, args.ablate) for r in df.itertuples()]
    classes = sorted(df.llm_name.unique())
    label_of = {c: i for i, c in enumerate(classes)}

    # vocabulary comes from the training split only
    counts = Counter(t for toks in df[df.split == "train"].tokens for t in toks)
    itos = ["<pad>", "<unk>"] + [
        w for w, c in counts.most_common(MAX_VOCAB) if c >= MIN_FREQ
    ]
    stoi = {w: i for i, w in enumerate(itos)}

    def make_loader(split, shuffle):
        part = df[df.split == split]
        seqs = [
            torch.tensor([stoi.get(t, UNK) for t in toks] or [UNK])
            for toks in part.tokens
        ]
        labels = [label_of[c] for c in part.llm_name]
        return DataLoader(
            TextDataset(seqs, labels),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    train_dl, val_dl, test_dl = (
        make_loader("train", True),
        make_loader("val", False),
        make_loader("test", False),
    )

    Model = TextCNN if args.model == "cnn" else TextLSTM
    model = Model(len(itos), len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"{args.model} | mode={args.mode} | ablate={args.ablate} | vocab={len(itos)} | "
        f"params={sum(p.numel() for p in model.parameters()):,} | device={device}"
    )

    best_val, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_p, tr_g = run_epoch(model, train_dl, device, optimizer)
        _, va_p, va_g = run_epoch(model, val_dl, device)
        tr_acc, va_acc = accuracy_score(tr_g, tr_p), accuracy_score(va_g, va_p)
        print(
            f"epoch {epoch:2d} | loss {tr_loss:.3f} | train acc {tr_acc:.3f} | val acc {va_acc:.3f}"
        )
        if va_acc > best_val:
            best_val, best_state, best_epoch, bad = (
                va_acc,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(
        best_state
    )  # test set is used once, with the best-on-val model
    _, te_p, te_g = run_epoch(model, test_dl, device)
    result = {
        "model": args.model,
        "mode": args.mode,
        "ablate": args.ablate,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_acc": best_val,
        "test_acc": accuracy_score(te_g, te_p),
        "test_macro_f1": f1_score(te_g, te_p, average="macro"),
        "classes": classes,
        "confusion_matrix": confusion_matrix(te_g, te_p).tolist(),
    }
    print(
        f"\nTEST acc {result['test_acc']:.3f} | macro-F1 {result['test_macro_f1']:.3f}"
    )
    print(classes)
    print(np.array(result["confusion_matrix"]))

    RESULTS.mkdir(exist_ok=True)
    tag = "" if args.ablate == "none" else f"_{args.ablate}"
    out = RESULTS / f"{args.model}_{args.mode}{tag}_seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
