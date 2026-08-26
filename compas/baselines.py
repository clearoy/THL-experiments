"""Traditional-ML baselines on the COMPAS split PolicyInduction will use.

Fits logistic regression, random forest, and gradient boosting on the exact
train/test split written by prepare_data.py, so the comparison against
PolicyInduction is apples-to-apples rather than against published numbers from
a different split.

Each model is fitted twice:

  full      - on all training rows. Comparable to the published literature.
  subsample - on the first `--n-train` rows only, matching what PolicyInduction
              can afford at LLM prices. This is the fair comparison: a method
              trained on 500 rows should not be measured against one trained on
              4,320.

Run:
    python experiments/compas/baselines.py
    python experiments/compas/baselines.py --n-train 500 --n-test 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
TARGET = "two_year_recid"


def make_models(seed: int) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def build_pipeline(model: Any, X: pd.DataFrame) -> Pipeline:
    """One-hot the categoricals, scale the numerics, then fit `model`."""
    cat = [c for c in X.columns if X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ]
    )
    return Pipeline([("pre", pre), ("clf", model)])


def score(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        # Accuracy is the headline metric here: COMPAS is ~45% positive, and
        # the rule-list literature reports accuracy. (VCBench used F0.5 because
        # it is ~9% positive and precision-sensitive.)
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run(n_train: int, n_test: int, seed: int) -> None:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    features = [c for c in train.columns if c not in (TARGET, "label")]

    test_eval = test.head(n_test) if n_test else test
    X_test, y_test = test_eval[features], test_eval[TARGET]

    regimes = {"full": train}
    if n_train and n_train < len(train):
        regimes["subsample"] = train.head(n_train)

    results: Dict[str, Any] = {
        "n_test": int(len(test_eval)),
        "features": features,
        "majority_class_accuracy": float(1 - y_test.mean()),
        "positive_rate_test": float(y_test.mean()),
        "regimes": {},
    }

    for regime, tr in regimes.items():
        X_tr, y_tr = tr[features], tr[TARGET]
        block: Dict[str, Any] = {"n_train": int(len(tr))}
        print(f"\n=== {regime}  (n_train={len(tr)}, n_test={len(test_eval)}) ===")
        for name, model in make_models(seed).items():
            pipe = build_pipeline(model, X_tr).fit(X_tr, y_tr)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            m = score(y_test, y_pred, y_prob)
            block[name] = m
            print(
                f"  {name:22s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                f"auc={m['roc_auc']:.4f}  p={m['precision']:.4f}  r={m['recall']:.4f}"
            )
        results["regimes"][regime] = block

    print(f"\n  {'majority class (all NO)':22s} acc={results['majority_class_accuracy']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-train",
        type=int,
        default=500,
        help="Rows for the subsample regime, matching the PolicyInduction run.",
    )
    p.add_argument("--n-test", type=int, default=1000, help="0 = full test set.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
