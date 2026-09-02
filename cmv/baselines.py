"""Word-count and bag-of-words baselines on the CMV pair task.

Both are fitted POINTWISE (one row per path-unit) and evaluated PAIRWISE: score
both members of a heldout pair, predict the higher-scoring one. That is the same
protocol run_policy_induction.py uses, so the numbers are comparable.

Free -- no API calls. Run this before any LLM spend.

Run:
    python experiments/cmv/baselines.py
    python experiments/cmv/baselines.py --n-train-units 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
CONDITIONS = ("root_reply", "full_path", "root_truncated")


def pairwise_accuracy(df: pd.DataFrame, score_col: str) -> float:
    """Fraction of pairs where the delta-winning unit scores higher.

    Ties count as 0.5, the expected value under a random tie-break. This is what
    forces the root_truncated word-count baseline to exactly 0.500 rather than
    to whichever side pandas happens to order first.
    """
    w = df.pivot(index="pair_id", columns="side", values=score_col)
    wins = (w["positive"] > w["negative"]).sum()
    ties = (w["positive"] == w["negative"]).sum()
    return float((wins + 0.5 * ties) / len(w))


def sample_units(train: pd.DataFrame, n_units: int, seed: int) -> pd.DataFrame:
    """Sample whole PAIRS, so the pointwise training set stays 50/50 balanced."""
    if not n_units or n_units >= len(train):
        return train
    pair_ids = train["pair_id"].drop_duplicates()
    n_pairs = max(1, n_units // 2)
    chosen = pair_ids.sample(n=min(n_pairs, len(pair_ids)), random_state=seed)
    return train[train["pair_id"].isin(chosen)].reset_index(drop=True)


def build_text(df: pd.DataFrame, condition: str, with_op: bool) -> pd.Series:
    """The unit's text as the model sees it: OP + argument, or argument alone."""
    arg = df[f"arg_{condition}"]
    if not with_op:
        return arg
    return df["op_title"] + "\n\n" + df["op_text"] + "\n\n" + arg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-train-units",
        type=int,
        default=0,
        help="Units to fit on; 0 = all. Set to 300 to match the LLM budget.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    fit_on = sample_units(train, args.n_train_units, args.seed)

    print(f"train: {len(fit_on)} units ({fit_on['pair_id'].nunique()} pairs)")
    print(f"eval : {len(heldout)} units ({heldout['pair_id'].nunique()} pairs)\n")

    results: Dict[str, Dict[str, float]] = {}
    hdr = f"{'condition':16s} {'word_count':>11s} {'bow_arg_only':>13s} {'bow_op+arg':>11s}"
    print(hdr)
    print("-" * len(hdr))

    for cond in CONDITIONS:
        row: Dict[str, float] = {}

        # 1. Longer argument wins. The confound the whole design is guarding against.
        h = heldout.copy()
        h["score"] = h[f"words_{cond}"]
        row["word_count"] = pairwise_accuracy(h, "score")

        # 2. TF-IDF + logistic regression, pointwise fit -> pairwise eval.
        for tag, with_op in (("bow_arg_only", False), ("bow_op_arg", True)):
            pipe = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                    ("clf", LogisticRegression(max_iter=2000, random_state=args.seed)),
                ]
            )
            pipe.fit(build_text(fit_on, cond, with_op), fit_on["delta"])
            h = heldout.copy()
            h["score"] = pipe.predict_proba(build_text(heldout, cond, with_op))[:, 1]
            row[tag] = pairwise_accuracy(h, "score")

        results[cond] = row
        flag = ""
        if cond == "root_truncated":
            flag = "   <- word_count must be 0.5000"
        print(
            f"{cond:16s} {row['word_count']:11.4f} {row['bow_arg_only']:13.4f} "
            f"{row['bow_op_arg']:11.4f}{flag}"
        )

    if abs(results["root_truncated"]["word_count"] - 0.5) > 1e-9:
        raise SystemExit(
            "\nFATAL: root_truncated word-count baseline is not exactly 0.5000. "
            "Truncation is broken; every downstream result would be uninterpretable."
        )

    print("\nReference points:")
    print("  59.6%  word count on root_reply (expected)")
    print("  65.1%  Tan et al. 2016 interplay features")
    print("  64.53% Labruna et al. 2026, best LLM result")
    print(f"\n  n={heldout['pair_id'].nunique()} pairs -> SE ~1.7pp; "
          "gaps under ~4pp are not distinguishable.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baselines.json"
    out.write_text(
        json.dumps(
            {
                "n_train_units": int(len(fit_on)),
                "n_eval_pairs": int(heldout["pair_id"].nunique()),
                "seed": args.seed,
                "conditions": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
