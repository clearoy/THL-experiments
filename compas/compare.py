"""Print the baseline vs PolicyInduction comparison table.

Reads results/baselines.json and (if present) results/policy_induction.json,
and renders one markdown table. Run after baselines.py and
run_policy_induction.py.

Run:
    python experiments/compas/compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
COLS = ("accuracy", "f1", "precision", "recall")


def fmt(row: dict) -> str:
    cells = " | ".join(f"{row.get(c, float('nan')):.4f}" for c in COLS)
    auc = f"{row['roc_auc']:.4f}" if "roc_auc" in row else "--"
    return f" {cells} | {auc} |"


def main() -> None:
    base_path = RESULTS_DIR / "baselines.json"
    if not base_path.exists():
        raise SystemExit("results/baselines.json not found. Run baselines.py first.")
    base = json.loads(base_path.read_text())

    lines = [
        f"COMPAS -- test set n={base['n_test']}, "
        f"{base['positive_rate_test']:.1%} positive",
        "",
        "| Model | Train rows | Accuracy | F1 | Precision | Recall | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]

    for regime, block in base["regimes"].items():
        n_train = block["n_train"]
        for name, m in block.items():
            if name == "n_train":
                continue
            label = f"{name} ({regime})"
            lines.append(f"| {label} | {n_train} |{fmt(m)}")

    pi_paths = sorted(RESULTS_DIR.glob("policy_induction_*.json"))
    if pi_paths:
        for p in pi_paths:
            pi = json.loads(p.read_text())
            run = p.stem.replace("policy_induction_", "")
            label = f"**PolicyInduction** ({run})"
            if pi.get("n_scored", 0) != pi.get("n_test", 0):
                label += f" [{pi['n_scored']}/{pi['n_test']} scored]"
            lines.append(f"| {label} | {pi['n_train']} |{fmt(pi)}")
    else:
        lines.append("| *PolicyInduction* | -- | *not run yet* | | | | |")

    lines += [
        f"| majority class (all NO) | -- | {base['majority_class_accuracy']:.4f} "
        "| -- | -- | -- | -- |",
        "",
        "Published reference: logistic regression on COMPAS is typically reported "
        "at 65-68% accuracy, ROC-AUC 0.70-0.73.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
