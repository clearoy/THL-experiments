# COMPAS benchmark for PolicyInduction

Benchmarks `PolicyInduction` against traditional ML on a structured dataset with
published baselines, so its performance can be read against something other than
itself.

**Why COMPAS.** Few free-text benchmarks exist for LLM-based classifiers, so this
uses a tabular dataset instead. COMPAS is the standard evaluation set for
*interpretable rule-learning* methods — CORELS, Bayesian Rule Lists, Falling Rule
Lists — which produce human-readable rules just as PolicyInduction does. That
makes the comparison like-for-like, not merely accuracy-vs-black-box.

No prose conversion is involved. `_render_sample()` already flattens each row to
`column: value` lines, so PolicyInduction and the sklearn baselines receive the
same information, and any difference is attributable to the method rather than to
a rewriting step.

## Results

All models evaluated on the same test set: n=1,852, 45.5% positive.

| Model | Train rows | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | 4,320 | 0.6830 | 0.6268 | 0.6753 | 0.5848 | 0.7339 |
| gradient_boosting | 4,320 | 0.6809 | 0.6363 | 0.6611 | 0.6133 | 0.7401 |
| random_forest | 4,320 | 0.6409 | 0.5887 | 0.6150 | 0.5647 | 0.6798 |
| logistic_regression | 500 | **0.6733** | 0.5619 | 0.7212 | 0.4603 | **0.7274** |
| **PolicyInduction** | 500 | **0.6641** | 0.5940 | 0.6604 | 0.5397 | **0.7156** |
| gradient_boosting | 500 | 0.6544 | 0.5784 | 0.6504 | 0.5208 | 0.7043 |
| random_forest | 500 | 0.6339 | 0.5789 | 0.6076 | 0.5528 | 0.6646 |
| majority class (all NO) | — | 0.5448 | — | — | — | — |

**Headline.** At equal training size (500 rows), PolicyInduction **beats random
forest and gradient boosting** on both accuracy and ROC-AUC, and sits **0.9
accuracy points / 1.2 AUC points behind logistic regression** — parity within
noise, while producing human-readable rules the other models cannot.

ROC-AUC is the fairest comparison, being threshold-independent. Logistic
regression at ~0.68 accuracy / ~0.73 AUC matches the published COMPAS range
(65–68%, AUC 0.70–0.73), confirming the pipeline is correctly configured.


### Two settings that decide whether this works

Both were established by an earlier run that scored **below the majority-class
baseline**. `--samples-per-batch` is set in the command below; `beta` is set in
`run_policy_induction.py`.

- **`beta=0.5`, not 1.0 — calibration.** F-beta ignores true negatives entirely,
  which is the wrong metric family for a ~45%-positive dataset. F1 specifically
  is maximised by over-predicting the positive class (trivial always-YES scores
  F1 = 0.625 here), which drove the threshold to 0.32 and 89% predicted-YES.
  Measured offline on saved policy scores, `beta=0.5` reproduces
  accuracy-optimal and balanced-accuracy-optimal threshold selection exactly —
  so it fixes the operating point without modifying `_fit_weights`.
- **`--samples-per-batch 20`, not 10 — capability.** At 10 with
  `class_ratio=(1,1)`, each generation round sees only **5 positive examples**,
  and the LLM wrote over-specified conjunctions ("female defendants under 25
  charged with a felony") matching almost nobody — 6 of 10 policies never fired
  once across 1,850 samples, leaving the regression ~4 usable features. At 20
  (10 YES + 10 NO per round) the policies generalise, and 10 of 15 fire in a
  healthy 4.6–47.9% range.

They are not interchangeable. All four training metrics (F1, F0.5, accuracy,
balanced accuracy) produce **identical ROC-AUC** on a given set of policy scores,
because AUC depends on feature quality rather than the cut point — so `beta`
cannot fix bad policies, and batch size cannot fix a bad threshold.

**Do not report F1 as the headline on this dataset.** The failing configuration
scored *higher* F1 (0.6391) than the working one (0.5940) while being worse by
every other measure.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Download, filter, split → `data/{train,test}.parquet`, `data/split_meta.json` |
| `baselines.py` | LR / RF / GB on the same split → `results/baselines.json`, `results/predictions_baselines_*.csv` |
| `run_policy_induction.py` | Fit + predict → `results/policy_induction_{run}.json`, `_predictions.csv`, and model artifacts in `{run}/` |
| `compare.py` | Renders the comparison table |

Per-row detail is saved for every model: probability, prediction, and threshold
per test sample, plus one column per policy for PolicyInduction. The metrics JSON
also carries a 19-point `threshold_sweep`, so the operating point can be
re-examined with no further API calls.

## Running

```bash
python experiments/compas/prepare_data.py
python experiments/compas/baselines.py --n-train 500
python experiments/compas/run_policy_induction.py --n-train 500 \
    --samples-per-batch 20 --outdir run01
python experiments/compas/compare.py
```

Both scripts default to the **full 1,852-row test set**. If you override
`--n-test`, you must pass the same value to both or the comparison is invalid.
Add `--dry-run` to `run_policy_induction.py` for a zero-cost estimate first. Only
that script costs money (~4,700 calls, ~6 min at `--concurrency 3`).

Re-running the same `--outdir` **resumes from its checkpoint** rather than
starting fresh — delete the run directory first for a genuinely new run.

## Data preparation decisions

**Row filter.** The raw CSV has 7,214 rows; ProPublica's published filter reduces
it to **6,172**, the subset most papers use. `prepare_data.py` applies it and
warns if the count drifts (which would mean the upstream CSV changed).

**Features (8).** `sex`, `age_years`, `age_group`, `juvenile_felony_convictions`,
`juvenile_misdemeanor_convictions`, `juvenile_other_offenses`,
`prior_offense_count`, `current_charge_severity` — the set used in the rule-list
literature.

Column names are mapped from the raw COMPAS abbreviations to readable forms
(`c_charge_degree: F` → `current_charge_severity: Felony`). This is a
deterministic dict lookup, not an LLM rewrite: no added variance, fully
reproducible. It is **neutral for the sklearn baselines** — one-hot encoding
yields identical features either way (verified: all metrics bit-identical) — and
removes an LLM-only handicap, since PolicyInduction renders rows as
`column: value` text where the column name carries the semantics.
`--raw-names` restores the original codes.

**Excluded, deliberately:**

- *Leakage* — `decile_score`, `score_text`, `v_decile_score` are COMPAS's own
  prediction of the very outcome being predicted. `is_recid`, `r_*`,
  `violent_recid`, `start/end/event` are the outcome or only knowable after it.
- *PII* — `name`, `first`, `last`, `dob` are real identities and must not reach
  an LLM API.
- *`race`* — off by default, available via `--include-race`. CORELS reports no
  optimal rule list selected a race antecedent. Running both ways is a legitimate
  experiment: PolicyInduction emits readable policies, so you can *see* whether
  race is used.

`prepare_data.py` asserts no leakage/PII column reaches the feature list rather
than relying on the list being correct by inspection.

**Split.** Stratified 70/30, `random_state=0` → 4,320 train / 1,852 test, both
45.5% positive.


## Known gaps

- **Single run.** Prior work on this codebase found two runs of an *identical*
  configuration differing by ~0.08 precision and ~0.15 recall on LLM sampling
  noise alone (generation runs at `temperature=1.0`, and `random_state` seeds
  only sample selection, not the LLM). The 0.7156 AUC is indicative, not
  conclusive — ≥3 runs are needed for an error bar.
- **`beta` and `--samples-per-batch` were changed together**, so the improvement
  over the failing configuration is not a clean ablation. The offline metric test
  isolates them (β cannot move AUC), but a run at `beta=1.0,
  --samples-per-batch 20` would settle it directly.
- **Target is re-arrest, not reconviction**, so it reflects policing patterns as
  well as behaviour — a caveat discussed extensively in the criminology and
  algorithmic-fairness literature.
- **5 of 15 policies still never fire**, so their scoring calls are wasted. A
  fire-rate warning during fit would catch this earlier.
- **`c_charge_desc` unused.** 389 free-text charge descriptions with a long
  singleton tail — the one column where semantic understanding could plausibly
  beat one-hot encoding, and the natural next experiment. Both methods should be
  run with and without it, since it is extra information the baselines do not
  currently receive.
