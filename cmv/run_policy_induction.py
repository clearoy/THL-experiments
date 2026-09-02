"""PolicyInduction on the CMV pair task: pointwise fit, pairwise evaluation.

Fits on ~300 individual path-units (one row each, label delta/no_delta), then
evaluates by scoring both members of a heldout pair and picking the higher.

The two members are never shown side by side. That formulation carries roughly
50pp of positional bias on this data, which would swamp any real signal.

Also runs a --control mode: a single generic "is this persuasive" prompt with no
induced rules. If the rule set does not beat the control, the induction added
nothing and the rules are decorative.

Run:
    python experiments/cmv/run_policy_induction.py --dry-run
    python experiments/cmv/run_policy_induction.py --condition root_reply --seed 0
    python experiments/cmv/run_policy_induction.py --control --condition root_reply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from think_reason_learn.core.llms import GoogleChoice  # noqa: E402
from think_reason_learn.policy_induction import (  # noqa: E402
    PolicyInduction,
    WeightTrainerConfig,
)
from cache_llm import CachingLLM  # noqa: E402

CONDITIONS = ("root_reply", "full_path", "root_truncated")

TASK_DESCRIPTION = (
    "On the /r/ChangeMyView subreddit, a user posts a view they hold and invites "
    "others to change it. If a reply succeeds in changing the poster's mind, the "
    "poster awards it a delta.\n\n"
    "Each sample contains:\n"
    "- op_title and op_text: the original poster's stated view and reasoning.\n"
    "- argument: one reply arguing against that view.\n\n"
    "Answer YES if this argument succeeded in changing the original poster's "
    "mind and earned a delta, NO if it did not. Judge how the argument is "
    "constructed and how it engages with the specific view in op_text. Argument "
    "length alone is not a reliable signal and should not drive the answer."
)

CONTROL_POLICY = (
    "This argument is persuasive enough to change the original poster's mind "
    "and earn a delta."
)


def thl_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_frame(df: pd.DataFrame, condition: str, op_words: int) -> pd.DataFrame:
    """Columns the LLM sees, rendered by _render_sample as `column: value`."""
    op = df["op_text"]
    if op_words:
        op = op.str.split().str[:op_words].str.join(" ")
    return pd.DataFrame(
        {"op_title": df["op_title"].values, "op_view": op.values,
         "argument": df[f"arg_{condition}"].values},
        index=df.index,
    )


def sample_units(train: pd.DataFrame, n_units: int, seed: int) -> pd.DataFrame:
    """Sample whole pairs so the pointwise training set is exactly 50/50."""
    pair_ids = train["pair_id"].drop_duplicates()
    n_pairs = max(1, n_units // 2)
    chosen = pair_ids.sample(n=min(n_pairs, len(pair_ids)), random_state=seed)
    return train[train["pair_id"].isin(chosen)].reset_index(drop=True)


def pairwise_accuracy(scores: pd.DataFrame) -> Dict[str, float]:
    """Score both members of each pair, take the higher. Ties count 0.5."""
    w = scores.pivot(index="pair_id", columns="side", values="probability")
    w = w.dropna()
    wins = (w["positive"] > w["negative"]).sum()
    ties = (w["positive"] == w["negative"]).sum()
    return {
        "pairwise_accuracy": float((wins + 0.5 * ties) / len(w)),
        "n_pairs_scored": int(len(w)),
        "n_ties": int(ties),
    }


async def main_async(args: argparse.Namespace) -> None:
    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    if args.n_eval_pairs:
        keep = heldout["pair_id"].drop_duplicates().head(args.n_eval_pairs)
        heldout = heldout[heldout["pair_id"].isin(keep)].reset_index(drop=True)

    fit_on = sample_units(train, args.n_train_units, args.seed)
    X_train = build_frame(fit_on, args.condition, args.op_words)
    y_train = fit_on["label"].tolist()
    X_eval = build_frame(heldout, args.condition, args.op_words)

    tag = f"{'control' if args.control else 'rules'}_{args.condition}_seed{args.seed}"
    outdir = HERE / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"condition : {args.condition}")
    print(f"mode      : {'CONTROL (1 generic prompt, no induction)' if args.control else 'induced rules'}")
    print(f"train     : {len(X_train)} units ({fit_on['pair_id'].nunique()} pairs, "
          f"{y_train.count('YES')} YES)")
    print(f"eval      : {len(X_eval)} units ({heldout['pair_id'].nunique()} pairs)")

    cache = CachingLLM(enabled=not args.no_cache)
    pi = PolicyInduction(
        gen_llmc=[GoogleChoice(model=args.gen_model)],
        predict_llmc=[GoogleChoice(model=args.predict_model)],
        # beta=1.0: the pointwise labels are exactly 50/50 by construction, so
        # precision and recall carry equal weight. (COMPAS used 0.5 for a
        # different reason -- see that experiment's README.)
        config=WeightTrainerConfig(beta=1.0, penalty="l1", cv_folds=args.cv_folds),
        max_policy_length=args.max_policies,
        class_ratio=(1.0, 1.0),
        max_samples_as_context=args.samples_per_batch,
        max_gen_batches=args.max_gen_batches,
        policy_batch_size=args.policy_batch_size,
        llm_semaphore_limit=args.concurrency,
        save_path=outdir,
        name=f"cmv_{tag}",
        confirm_requests=False,
        random_state=args.seed,
        _llm=cache,
    )

    if args.dry_run:
        await pi.set_task(
            task_description=TASK_DESCRIPTION,
            instructions_template="STUB (dry run). Max <max_policy_length> policies.",
        )
        pi._set_data(X_train, y_train)
        est = pi._estimate_fit_requests()
        n_pol = 1 if args.control else args.max_policies
        n_pred = len(X_eval) * -(-n_pol // args.policy_batch_size)
        print("\nEstimated requests (no calls made):")
        if not args.control:
            for k, v in est.items():
                print(f"  {k}: ~{v}")
        print(f"  {args.predict_model} (predict): ~{n_pred}")
        total = n_pred + (0 if args.control else sum(est.values()))
        print(f"  TOTAL: ~{total}")
        print("\nDry run complete. Drop --dry-run to execute.")
        return

    if args.control:
        # No generation, no weight fitting from induced rules: one fixed policy,
        # and the raw YES-rate becomes the score.
        await pi.set_task(
            task_description=TASK_DESCRIPTION,
            instructions_template="Unused in control mode. <max_policy_length>",
        )
        pi._set_data(X_train, y_train)
        pi._policy_memory = pd.DataFrame(
            {"policy": [CONTROL_POLICY], "predictions": [None]}
        )
        await pi._score_policies()
        pi._fit_weights()
    else:
        await pi.set_task(task_description=TASK_DESCRIPTION)
        await pi.fit(X_train, y_train)
    pi.save()

    rows, vectors = [], []
    async for idx, vec, pred, _tc in pi.predict(X_eval):
        rows.append(idx)
        vectors.append(vec)

    V = np.array(vectors, dtype=float)
    scores = heldout.loc[rows, ["pair_id", "side", "delta"]].copy()
    scores["probability"] = pi.lr.predict_proba(V)[:, 1]
    for j, name in enumerate(pi._feature_order_):
        scores[f"policy_{name}"] = V[:, j]

    metrics = pairwise_accuracy(scores)
    metrics.update(
        {
            "condition": args.condition,
            "mode": "control" if args.control else "rules",
            "seed": args.seed,
            "n_train_units": len(X_train),
            "n_units_scored": len(scores),
            "n_units_expected": len(X_eval),
            "n_policies_generated": int(len(pi._feature_order_)),
            "n_policies_nonzero": int(np.count_nonzero(pi.lr.coef_[0])),
            "pointwise_accuracy": float((scores["probability"] >= pi.threshold).astype(int)
                                        .eq(scores["delta"]).mean()),
            "threshold": float(pi.threshold),
            "validation_result": pi.validation_result,
            "thl_commit": thl_commit(),
            "cache": cache.stats,
            "config": {
                "gen_model": args.gen_model, "predict_model": args.predict_model,
                "max_policies": args.max_policies, "samples_per_batch": args.samples_per_batch,
                "max_gen_batches": args.max_gen_batches,
                "policy_batch_size": args.policy_batch_size, "op_words": args.op_words,
            },
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(metrics, indent=2))
    scores.to_csv(RESULTS_DIR / f"{tag}_scores.csv", index=False)

    print(f"\n=== {tag} ===")
    print(f"  pairwise accuracy : {metrics['pairwise_accuracy']:.4f}  "
          f"({metrics['n_pairs_scored']} pairs, {metrics['n_ties']} ties)")
    print(f"  pointwise accuracy: {metrics['pointwise_accuracy']:.4f}")
    print(f"  policies          : {metrics['n_policies_nonzero']}/"
          f"{metrics['n_policies_generated']} non-zero")
    print(f"  cache             : {cache.stats}")
    cache.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, default="root_reply")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train-units", type=int, default=300)
    p.add_argument("--n-eval-pairs", type=int, default=0, help="0 = all 807.")
    p.add_argument("--control", action="store_true",
                   help="One generic prompt, no induced rules.")
    p.add_argument("--gen-model", default="gemini-3.5-flash",
                   help="Strong model: writes the policies.")
    p.add_argument("--predict-model", default="gemini-2.5-flash-lite",
                   help="Cheap model: evaluates each rule against each unit.")
    p.add_argument("--max-policies", type=int, default=15)
    p.add_argument("--samples-per-batch", type=int, default=20)
    p.add_argument("--max-gen-batches", type=int, default=7)
    p.add_argument("--policy-batch-size", type=int, default=15)
    p.add_argument("--op-words", type=int, default=200,
                   help="Truncate op_text to N words (0 = full). Controls cost.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
