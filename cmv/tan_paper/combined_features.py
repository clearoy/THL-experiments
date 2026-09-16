"""Does combining Tan-style features with induced policies improve heldout
accuracy, on the identical 500-pair training set PolicyInduction used?

Zero new API calls. Everything needed is already on disk:
    - training pairs + labels     : cmv/runs/.../data.parquet (verified to
                                     match a fresh build_pair_frame+sample_pairs
                                     draw row-for-row before this script existed)
    - training policy answers     : cmv/runs/.../policy_predictions.parquet
                                     (sample_index 0..499 maps positionally to
                                     data.parquet's rows -- confirmed directly)
    - heldout policy answers      : cmv/results/..._scores.csv (policy_0..14,
                                     RAW single-order values, plus pair_id and
                                     a_is_positive)
    - heldout argument texts      : cmv/data/units_heldout.parquet, joined back
                                     via pair_id + a_is_positive so "argument_A"
                                     means the same physical reply the policy
                                     features were computed against

No swap anywhere, per current standing instruction: every feature (Tan and
policy alike) is computed once, in the single fixed order the original run
used. This is NOT the swap-averaged 0.6241 number quoted elsewhere for
PolicyInduction, and not the untruncated-OP, no-intercept, all-3456-pair
number quoted for the standalone Tan reimplementation. All three models
below -- Tan alone, policies alone, combined -- are refit under one
identical protocol so the comparison isolates the feature set, not the
fitting procedure:

    - fit_intercept=True for all three (policy features are not
      antisymmetric, so forcing no intercept would only handicap Tan)
    - GroupKFold(5) by original post for C selection, all three
    - op_view truncated to 300 words, matching what PolicyInduction itself
      consumed -- NOT the full text the standalone Tan script used
    - same C grid across all three

So the "Tan alone" and "policies alone" numbers here are expected to differ
slightly from the headline numbers quoted in the two READMEs. That is by
design: this script isolates one variable (does adding policy features
help) and holds everything else fixed.

Run:
    python experiments/cmv/tan_paper/combined_features.py --dry-run
    python experiments/cmv/tan_paper/combined_features.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
CMV_DIR = HERE.parent
DATA_DIR = CMV_DIR / "data"
RESULTS_DIR = CMV_DIR / "results"
RUN_DIR = CMV_DIR / "runs" / "rules_paired_root_reply_n500_gen35flash_pred35flash_seed0"
SCORES_CSV = RESULTS_DIR / "rules_paired_root_reply_n500_gen35flash_pred35flash_seed0_scores.csv"

sys.path.insert(0, str(CMV_DIR))
sys.path.insert(0, str(HERE))
import run_policy_induction as rpi  # noqa: E402
import reproduce_features as tf  # noqa: E402

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def tan_features(arg_a: pd.Series, arg_b: pd.Series, op: pd.Series) -> np.ndarray:
    ip_a = np.stack([tf.interplay(a, o) for a, o in zip(arg_a, op)])
    ip_b = np.stack([tf.interplay(b, o) for b, o in zip(arg_b, op)])
    st_a = np.stack([tf.style(a) for a in arg_a])
    st_b = np.stack([tf.style(b) for b in arg_b])
    return np.hstack([ip_a - ip_b, st_a - st_b])


def load_train():
    """Reconstruct the exact 500-pair training set, verified identical to
    data.parquet row-for-row, so pair_id (and hence op_name) is recoverable."""
    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    X_all, y_all, _ = rpi.build_pair_frame(train, "root_reply", 300, 0)
    X_500, y_500 = rpi.sample_pairs(X_all, y_all, 500, 0)

    saved = pd.read_parquet(RUN_DIR / "data.parquet")
    check_cols = ["op_title", "op_view", "argument_A", "argument_B"]
    assert X_500.reset_index(drop=True)[check_cols].equals(saved[check_cols]), (
        "Reconstructed training set no longer matches data.parquet -- the "
        "positional alignment this script depends on is broken."
    )

    op_name = train.groupby("pair_id").op_name.first().reindex(X_500.index)
    X_500 = X_500.assign(op_name=op_name.to_numpy())

    pred = pd.read_parquet(RUN_DIR / "policy_predictions.parquet")
    pred["policy_id"] = pred["policy_id"].astype(int)
    policy_mat = (
        pred.pivot(index="sample_index", columns="policy_id", values="pred")
        .sort_index()
        .sort_index(axis=1)
    )
    policy_mat = (policy_mat == "YES").astype(float).to_numpy()
    assert policy_mat.shape == (500, 15), f"unexpected policy matrix shape {policy_mat.shape}"

    y = (pd.Series(y_500).reset_index(drop=True) == "YES").to_numpy().astype(int)
    return X_500, y, policy_mat


def load_heldout():
    scores = pd.read_csv(SCORES_CSV)
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    wide = heldout.pivot(index="pair_id", columns="side", values="arg_root_reply")
    op_full = heldout.groupby("pair_id").op_text.first()
    op_300 = op_full.str.split().str[:300].str.join(" ")

    pos, neg = wide["positive"], wide["negative"]
    a_is_pos = scores.set_index("pair_id")["a_is_positive"].reindex(scores.pair_id).to_numpy()
    pids = scores.pair_id.to_numpy()

    arg_a = pd.Series(np.where(a_is_pos, pos.loc[pids].to_numpy(), neg.loc[pids].to_numpy()))
    arg_b = pd.Series(np.where(a_is_pos, neg.loc[pids].to_numpy(), pos.loc[pids].to_numpy()))
    op = pd.Series(op_300.loc[pids].to_numpy())

    policy_cols = [c for c in scores.columns if c.startswith("policy_")]
    policy_cols = sorted(policy_cols, key=lambda c: int(c.split("_")[1]))
    policy_mat = scores[policy_cols].to_numpy(dtype=float)
    y = (scores.label == "YES").to_numpy().astype(int)
    return arg_a, arg_b, op, policy_mat, y, pids


def fit_eval(X, y, groups, Xte, yte):
    gkf = GroupKFold(n_splits=5)
    best_c, best_cv = C_GRID[0], -np.inf
    for c in C_GRID:
        accs = []
        for tr, va in gkf.split(X, y, groups):
            lr = LogisticRegression(C=c, solver="liblinear", l1_ratio=1.0, random_state=0, max_iter=3000)
            lr.fit(X[tr], y[tr])
            accs.append((lr.predict(X[va]) == y[va]).mean())
        if np.mean(accs) > best_cv:
            best_c, best_cv = c, float(np.mean(accs))
    lr = LogisticRegression(C=best_c, solver="liblinear", l1_ratio=1.0, random_state=0, max_iter=3000)
    lr.fit(X, y)
    p = lr.predict_proba(Xte)[:, 1]
    acc = float(np.where(p == 0.5, 0.5, (p > 0.5) == yte).mean())
    return {"accuracy": acc, "cv_accuracy": best_cv, "C": best_c,
            "n_nonzero": int(np.count_nonzero(lr.coef_[0]))}, p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        print("Zero API calls -- everything needed is already on disk.")
        print("Would load the 500-pair training set + 15 cached policy answers,")
        print("compute 33 Tan features locally, refit three models, evaluate on")
        print("the 806 heldout pairs PolicyInduction already scored.")
        return

    Xtr, ytr, policy_tr = load_train()
    groups = Xtr.op_name.to_numpy()
    tan_tr = tan_features(Xtr.argument_A, Xtr.argument_B, Xtr.op_view)

    arg_a_te, arg_b_te, op_te, policy_te, yte, pids = load_heldout()
    tan_te = tan_features(arg_a_te, arg_b_te, op_te)

    print(f"train: {len(ytr)} pairs, {len(set(groups))} distinct OPs")
    print(f"heldout: {len(yte)} pairs\n")

    configs = {
        "Tan features alone (33)": (tan_tr, tan_te),
        "Policies alone (15)": (policy_tr, policy_te),
        "Combined (48)": (np.hstack([tan_tr, policy_tr]), np.hstack([tan_te, policy_te])),
    }

    rows = []
    probs = {}
    print(f"{'model':28s} {'heldout acc':>12s} {'cv acc':>8s} {'C':>7s} {'nonzero':>8s}")
    for name, (X, Xte) in configs.items():
        r, p = fit_eval(X, ytr, groups, Xte, yte)
        rows.append({"model": name, "n_features": X.shape[1], **r})
        probs[name] = p
        print(f"{name:28s} {r['accuracy']:12.4f} {r['cv_accuracy']:8.4f} "
              f"{r['C']:7g} {r['n_nonzero']:8d}")

    n = len(yte)
    combined_correct = np.where(probs["Combined (48)"] == 0.5, 0.5,
                                 (probs["Combined (48)"] > 0.5) == yte)
    print("\npaired comparison vs combined:")
    for name in ("Tan features alone (33)", "Policies alone (15)"):
        p = probs[name]
        c = np.where(p == 0.5, 0.5, (p > 0.5) == yte)
        d = combined_correct - c
        se = d.std(ddof=1) / np.sqrt(n)
        z = d.mean() / se if se > 0 else float("nan")
        print(f"  combined - {name:26s} diff={d.mean():+.4f}  SE={se:.4f}  z={z:+.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tan_plus_policy_root_reply_n500_seed0.json"
    out.write_text(json.dumps(
        {"n_train_pairs": len(ytr), "n_heldout_pairs": len(yte),
         "protocol": "fit_intercept=True, GroupKFold(5) by op_name, "
                     "op_view truncated to 300 words, no swap, single fixed order",
         "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
