"""Accuracy table, verbatim rule sets, and the Tan-feature mapping.

Reads whatever runs exist in results/ and renders:

  1. accuracy across the three conditions, rules vs control, vs the baselines
  2. every induced rule verbatim, with its weight
  3. which rules correspond to a feature Tan et al. 2016 report, and which
     are novel

The mapping is keyword-based and therefore a first pass, not a verdict --
`--verbose` prints the matched terms so each assignment can be checked by hand.

Run:
    python experiments/cmv/analyze.py
    python experiments/cmv/analyze.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RUNS_DIR = HERE / "runs"
CONDITIONS = ("root_reply", "full_path", "root_truncated")

REFERENCE = {
    "word count (this repo, root_reply)": 0.596,
    "Tan et al. 2016 interplay features": 0.651,
    "Labruna et al. 2026 best LLM": 0.6453,
}

# Features Tan et al. 2016 report as predictive. A rule matching one of these is
# a rediscovery; anything unmatched is a candidate novel signal.
TAN_FEATURES: Dict[str, List[str]] = {
    "word overlap with OP": [
        "overlap", "same word", "quote", "quoting", "echo", "restate", "mirror",
        "op's own", "poster's own", "their own words", "terminology",
    ],
    "links / evidence": [
        "link", "url", "source", "citation", "cite", "evidence", "study",
        "statistic", "data", "reference",
    ],
    "hedging / calibrated language": [
        "hedge", "hedging", "tentative", "qualifier", "may ", "might ", "could ",
        "perhaps", "uncertain", "acknowledg", "concede", "concession", "nuance",
    ],
    "examples / concreteness": [
        "example", "concrete", "specific instance", "anecdote", "illustrat",
        "scenario", "for instance", "analogy",
    ],
    "low arousal / calm tone": [
        "calm", "measured", "arousal", "neutral tone", "respectful", "polite",
        "non-confrontational", "aggressive", "hostile", "insult", "dismissive",
    ],
    "definite / assertive phrasing": [
        "definite", "assertive", "confident", "certainty", "absolute", "categorical",
    ],
    "length / detail": [
        "long", "length", "detailed", "elaborat", "word count", "brief", "short",
    ],
    "question-asking": [
        "question", "ask ", "asks ", "asking", "socratic", "clarif",
    ],
}


def classify(rule: str) -> Tuple[List[str], List[str]]:
    low = rule.lower()
    matched, terms = [], []
    for feat, kws in TAN_FEATURES.items():
        hit = [k for k in kws if k in low]
        if hit:
            matched.append(feat)
            terms.extend(hit)
    return matched, terms


def load_runs() -> List[dict]:
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name == "baselines.json":
            continue
        try:
            out.append({**json.loads(p.read_text()), "_file": p.stem})
        except Exception:
            pass
    return out


def rules_for(tag: str) -> List[Tuple[float, str]]:
    """Weighted rules from a run's report.md, ranked by |weight|."""
    rep = RUNS_DIR / tag / "report.md"
    if not rep.exists():
        return []
    body = rep.read_text()
    if "## Policies used by the model" not in body:
        return []
    sec = body.split("## Policies used by the model")[1]
    rows = re.findall(r"\|\s*\d+\s*\|\s*([+-][\d.]+)\s*\|\s*(.+?)\s*\|", sec)
    return [(float(w), t) for w, t in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="Show matched keywords.")
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        raise SystemExit("No runs in results/. Run run_policy_induction.py first.")

    # ── 1. accuracy table ─────────────────────────────────────────────────────
    print("=" * 78)
    print("1. PAIRWISE ACCURACY  (heldout, n=807 pairs unless noted; SE ~1.7pp)")
    print("=" * 78)

    bl_path = RESULTS_DIR / "baselines.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text())["conditions"]
        print(f"\n{'baseline':22s} " + " ".join(f"{c:>16s}" for c in CONDITIONS))
        for name in ("word_count", "bow_arg_only", "bow_op_arg"):
            cells = " ".join(f"{bl[c][name]:16.4f}" for c in CONDITIONS)
            print(f"{name:22s} {cells}")

    df = pd.DataFrame(
        [
            {"mode": r["mode"], "condition": r["condition"], "seed": r["seed"],
             # Runs predating the paired formulation carry no such field.
             "formulation": r.get("formulation", "pointwise"),
             "acc": r["pairwise_accuracy"], "n": r["n_pairs_scored"],
             "nz": r.get("n_policies_nonzero"), "gen": r.get("n_policies_generated")}
            for r in runs
        ]
    )
    print(f"\n{'PolicyInduction':22s} " + " ".join(f"{c:>16s}" for c in CONDITIONS))
    # Paired and pointwise are different experiments and must never be averaged
    # into one cell.
    for form, mode in [(f, m) for f in ("paired", "pointwise")
                       for m in ("rules", "control")]:
        sub = df[(df["mode"] == mode) & (df["formulation"] == form)]
        if sub.empty:
            continue
        cells = []
        for c in CONDITIONS:
            s = sub[sub["condition"] == c]["acc"]
            if s.empty:
                cells.append(f"{'--':>16s}")
            elif len(s) == 1:
                cells.append(f"{s.iloc[0]:16.4f}")
            else:
                cells.append(f"{s.mean():10.4f}±{s.std():.3f}")
        label = f"{form}/{mode} (n={len(sub)})"
        print(f"{label:22s} " + " ".join(cells))

    print("\nreference points:")
    for k, v in REFERENCE.items():
        print(f"  {v:.4f}  {k}")

    for form in ("paired", "pointwise"):
        f_df = df[(df["formulation"] == form) & (df["condition"] == "root_reply")]
        rules_acc = f_df[f_df["mode"] == "rules"]["acc"]
        ctrl_acc = f_df[f_df["mode"] == "control"]["acc"]
        if rules_acc.empty or ctrl_acc.empty:
            continue
        gap = rules_acc.mean() - ctrl_acc.mean()
        verdict = "induction added signal" if gap > 0.04 else \
                  "NOT distinguishable from the no-rule control"
        print(f"\n  [{form}] rules - control on root_reply: {gap:+.4f}  ->  {verdict}")
        print("  (a gap under ~4pp is within noise at n=807)")

    # ── 2 + 3. rules and the Tan mapping ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("2. INDUCED RULES (verbatim)   3. MAPPING TO TAN ET AL. 2016 FEATURES")
    print("=" * 78)

    for r in runs:
        if r["mode"] != "rules":
            continue
        tag = r["_file"]
        rules = rules_for(tag)
        if not rules:
            continue
        print(f"\n--- {tag}  (acc {r['pairwise_accuracy']:.4f}, "
              f"{r.get('n_policies_nonzero')}/{r.get('n_policies_generated')} non-zero) ---")
        novel = 0
        for w, text in rules:
            feats, terms = classify(text)
            tagstr = " | ".join(feats) if feats else "** NOVEL (no Tan feature matched) **"
            if not feats:
                novel += 1
            print(f"\n  [{w:+.4f}] {text}")
            print(f"      -> {tagstr}")
            if args.verbose and terms:
                print(f"         matched: {sorted(set(terms))}")
        print(f"\n  summary: {len(rules) - novel}/{len(rules)} map to a Tan feature, "
              f"{novel} novel")

    print("\nNote: the mapping is keyword-based, so it is a first pass for manual "
          "review\n      rather than a verdict. Use --verbose to see matched terms.")


if __name__ == "__main__":
    main()
