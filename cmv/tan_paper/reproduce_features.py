"""Reimplementation attempt of Tan et al. 2016's Section 4 features.

Not a verified reproduction -- a best-effort reimplementation with every
deviation from the paper documented below and printed at runtime. No API
calls, no cost: pure Python plus pandas/numpy/sklearn on data already on
disk.

WHAT THE PAPER SPECIFIES EXACTLY, implemented faithfully here:
    - Stopwords: "we remove stopwords as defined by Mallet's dictionary."
      mallet_stoplist.txt in this directory is the verbatim list from
      https://github.com/mimno/Mallet/blob/master/stoplists/en.txt (524
      words), fetched directly -- not reconstructed from memory.
    - Interplay features: word overlap between argument and OP, computed
      over three vocabularies (stopwords / content words / all words),
      four quantities each (shared count, shared/|argument|,
      shared/|OP|, Jaccard) = 12 features.
    - Fitting protocol: L1-regularised logistic regression, 5-fold CV,
      all pairs sharing an OP kept in the same fold (GroupKFold on
      op_name).

WHAT THE PAPER NAMES BUT DOES NOT FULLY SPECIFY, approximated here with
the approximation stated:
    - "Positive and negative lexicons from LIWC": LIWC is commercial and
      not available. Substituted with the Hu & Liu 2004 Opinion Lexicon
      (positive-words.txt / negative-words.txt, ~6,800 words total, a
      different and freely available sentiment lexicon). NOT equivalent
      to LIWC's category counts.
    - Hedging cues: the paper curates a list from two citations it does
      not reproduce. HEDGES below is this script's own list, not theirs.
    - The regularisation-strength grid: not given. C_GRID below is a
      standard geometric grid, not theirs.
    - Whether the two arguments' features are combined by difference or
      concatenation for the pairwise classifier: not stated. This script
      uses the DIFFERENCE, with no intercept, matching embedding_baseline.py
      -- an explicit design choice, not a documented replication.

NOT IMPLEMENTED AT ALL, and excluded rather than faked:
    - Arousal / valence / dominance (Warriner et al. 2013 norms) and
      concreteness (Brysbaert et al. 2014 norms): these are named,
      licensed-for-research word-norm datasets not bundled with this
      repo. No feature stands in for them.
    - Part-of-speech tags: no tagger is named in the paper, and none is
      installed in this project's environment (checked: no nltk, no
      spacy). The POS-tag comparison group is omitted, not approximated.

Everything else here (articles, pronouns, examples, links, quotations,
question marks, formatting, word/sentence/paragraph counts, type-token
ratio, Flesch-Kincaid, word entropy) is a plain reimplementation from the
paper's feature names, with no external dependency beyond what ships with
this repo's environment.

Run:
    python experiments/cmv/tan_paper/reproduce_features.py --dry-run
    python experiments/cmv/tan_paper/reproduce_features.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "cmv" / "data"
RESULTS_DIR = HERE.parents[1] / "cmv" / "results"
CONDITIONS = ("root_reply", "full_path", "root_truncated")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)  # not the paper's grid

# ── lexicons ─────────────────────────────────────────────────────────────────

STOPWORDS = set((HERE / "mallet_stoplist.txt").read_text().split())
POS_WORDS = set((HERE / "positive-words.txt").read_text().split())
NEG_WORDS = set((HERE / "negative-words.txt").read_text().split())

# This script's own list. Not the paper's -- see module docstring.
HEDGES = {
    "may", "might", "could", "perhaps", "possibly", "seems", "seem",
    "appears", "appear", "suggest", "suggests", "somewhat", "generally",
    "often", "sometimes", "arguably", "presumably", "likely", "unlikely",
    "probably", "tend", "tends", "apparently", "reportedly", "roughly",
    "approximately", "typically", "usually", "assume", "assuming", "guess",
    "think", "believe", "seemingly",
}
EXAMPLE_MARKERS = ("for example", "for instance", "e.g.", "such as")

_WORD_RE = re.compile(r"[a-zA-Z']+")
_SENT_RE = re.compile(r"[.!?]+")
_URL_RE = re.compile(r"https?://\S+|\[[^\]]+\]\(\S+\)")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _syllables(word: str) -> int:
    """Heuristic vowel-group count. Approximate -- no dictionary lookup."""
    n = len(_VOWEL_RE.findall(word))
    return max(1, n)


def interplay(text: str, op: str) -> np.ndarray:
    """12 features: (shared, shared/|text|, shared/|op|, jaccard) x 3 vocabs."""
    t, o = set(_words(text)), set(_words(op))
    out = []
    for vocab_filter in (
        lambda s: s & STOPWORDS,          # stopwords only
        lambda s: s - STOPWORDS,          # content words only
        lambda s: s,                       # all words
    ):
        a, b = vocab_filter(t), vocab_filter(o)
        shared = a & b
        union = a | b
        out.extend(
            [
                len(shared),
                len(shared) / len(a) if a else 0.0,
                len(shared) / len(b) if b else 0.0,
                len(shared) / len(union) if union else 0.0,
            ]
        )
    return np.array(out, dtype=float)


def style(text: str) -> np.ndarray:
    """~21 features. See module docstring for what is faithful vs substituted."""
    words = _words(text)
    n = len(words) or 1
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sent = len(sentences) or 1
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    n_syll = sum(_syllables(w) for w in words)
    counts = Counter(words)
    probs = np.array(list(counts.values())) / n
    entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    low = text.lower()

    return np.array(
        [
            len(words),
            n_sent,
            len(paragraphs),
            len(counts) / n,                                    # type-token ratio
            sum(len(w) for w in words) / n,                     # avg word length
            0.39 * (n / n_sent) + 11.8 * (n_syll / n) - 15.59,   # Flesch-Kincaid grade
            entropy,
            low.split().count("the") / n,                       # definite article rate
            (words.count("a") + words.count("an")) / n,         # indefinite article rate
            sum(words.count(w) for w in ("i", "me", "my", "mine", "myself")) / n,
            sum(words.count(w) for w in ("we", "us", "our", "ours", "ourselves")) / n,
            sum(words.count(w) for w in ("you", "your", "yours", "yourself")) / n,
            sum(words.count(w) for w in HEDGES) / n,
            sum(low.count(m) for m in EXAMPLE_MARKERS),
            len(_URL_RE.findall(text)),
            text.count(">") + text.count('"'),                  # quote markers
            text.count("?"),
            sum(1 for w in words if w in POS_WORDS) / n,
            sum(1 for w in words if w in NEG_WORDS) / n,
            len(re.findall(r"\*\*[^*]+\*\*", text)),             # bold
            sum(1 for p in text.split("\n") if p.strip().startswith(("- ", "* "))),
        ],
        dtype=float,
    )


N_INTERPLAY = 12
N_STYLE = 21


def build_pairs(df: pd.DataFrame, condition: str, seed: int) -> pd.DataFrame:
    """One row per pair. op_name is kept for GroupKFold; op_text is NOT
    truncated -- unlike the LLM runs, there is no per-call cost here."""
    wide = df.pivot(index="pair_id", columns="side", values=f"arg_{condition}").dropna()
    op = df.groupby("pair_id")[["op_name", "op_text"]].first().reindex(wide.index)

    n = len(wide)
    a_is_pos = np.zeros(n, dtype=bool)
    a_is_pos[: n // 2] = True
    np.random.default_rng(seed).shuffle(a_is_pos)

    pos, neg = wide["positive"].to_numpy(), wide["negative"].to_numpy()
    return pd.DataFrame(
        {
            "op_name": op["op_name"].to_numpy(),
            "op_text": op["op_text"].to_numpy(),
            "arg_A": np.where(a_is_pos, pos, neg),
            "arg_B": np.where(a_is_pos, neg, pos),
            "y": a_is_pos.astype(int),
        },
        index=wide.index,
    )


def featurise(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    interplay_a = np.stack([interplay(a, o) for a, o in zip(frame.arg_A, frame.op_text)])
    interplay_b = np.stack([interplay(b, o) for b, o in zip(frame.arg_B, frame.op_text)])
    style_a = np.stack([style(a) for a in frame.arg_A])
    style_b = np.stack([style(b) for b in frame.arg_B])
    return {
        "interplay": interplay_a - interplay_b,
        "style": style_a - style_b,
        "interplay + style": np.hstack([interplay_a - interplay_b, style_a - style_b]),
    }


def fit_eval_grouped(X, y, groups, Xte, yte) -> Dict[str, float]:
    """L1 logistic, no intercept, C chosen by 5-fold GroupKFold accuracy."""
    gkf = GroupKFold(n_splits=5)
    best_c, best_cv = C_GRID[0], -np.inf
    for c in C_GRID:
        accs = []
        for tr, va in gkf.split(X, y, groups):
            lr = LogisticRegression(C=c, solver="liblinear", l1_ratio=1.0, random_state=0,
                                     fit_intercept=False, max_iter=3000)
            lr.fit(X[tr], y[tr])
            accs.append((lr.predict(X[va]) == y[va]).mean())
        if np.mean(accs) > best_cv:
            best_c, best_cv = c, float(np.mean(accs))
    lr = LogisticRegression(C=best_c, solver="liblinear", l1_ratio=1.0, random_state=0,
                             fit_intercept=False, max_iter=3000)
    lr.fit(X, y)
    p = lr.predict_proba(Xte)[:, 1]
    acc = float(np.where(p == 0.5, 0.5, (p > 0.5) == yte).mean())
    return {"accuracy": acc, "cv_accuracy": best_cv, "C": best_c,
            "n_nonzero": int(np.count_nonzero(lr.coef_[0]))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, default="root_reply")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-pairs", type=int, default=0,
                   help="0 = all 3,456, matching the paper's protocol.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print("Deviations from the paper -- see module docstring for detail:")
    print("  * positive/negative lexicon : Hu & Liu 2004, NOT LIWC")
    print("  * hedge word list           : this script's own, NOT the paper's")
    print("  * arousal/valence/dominance/concreteness : NOT implemented")
    print("  * POS tags                  : NOT implemented (no tagger available)")
    print("  * A/B combination           : difference vector (paper unspecified)")
    print("  * C grid                    : this script's own (paper unspecified)")
    print()

    if args.dry_run:
        print(f"Would fit on {'all 3,456' if not args.n_train_pairs else args.n_train_pairs} "
              f"training pairs, evaluate on 807 heldout. Zero API calls; local only.")
        return

    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    tr = build_pairs(train, args.condition, args.seed)
    te = build_pairs(heldout, args.condition, args.seed)
    if args.n_train_pairs:
        rng = np.random.default_rng(args.seed)
        idx = np.concatenate([
            rng.permutation(np.where(tr.y == 1)[0])[: args.n_train_pairs // 2],
            rng.permutation(np.where(tr.y == 0)[0])[: args.n_train_pairs - args.n_train_pairs // 2],
        ])
        tr = tr.iloc[idx]

    print(f"condition : {args.condition}")
    print(f"train     : {len(tr)} pairs ({tr.op_name.nunique()} distinct OPs)")
    print(f"heldout   : {len(te)} pairs\n")

    Ftr, Fte = featurise(tr), featurise(te)
    ytr, yte, groups = tr.y.to_numpy(), te.y.to_numpy(), tr.op_name.to_numpy()

    rows = []
    print(f"{'features':22s} {'n_feat':>7s} {'heldout acc':>12s} {'cv acc':>8s} {'C':>7s} {'nonzero':>8s}")
    for name, X in Ftr.items():
        r = fit_eval_grouped(X, ytr, groups, Fte[name], yte)
        rows.append({"features": name, "n_features": X.shape[1], **r})
        print(f"{name:22s} {X.shape[1]:7d} {r['accuracy']:12.4f} {r['cv_accuracy']:8.4f} "
              f"{r['C']:7g} {r['n_nonzero']:8d}")

    print("\nreference points:")
    print("  0.6510  Tan et al. 2016, interplay features (quoted, not this script)")
    print("  0.5967  word count (this repo, verified reproduction)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"tan_features_{args.condition}_n{args.n_train_pairs or 'all'}_seed{args.seed}.json"
    out.write_text(json.dumps(
        {"condition": args.condition, "seed": args.seed,
         "n_train_pairs": len(tr), "n_heldout_pairs": len(te),
         "results": rows,
         "deviations": [
             "positive/negative lexicon: Hu & Liu 2004, not LIWC",
             "hedge list: this script's own",
             "arousal/valence/dominance/concreteness: not implemented",
             "POS tags: not implemented",
             "A/B combination: difference vector, paper unspecified",
             "C grid: this script's own, paper unspecified",
         ]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
