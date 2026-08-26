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

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Download, filter, split. Writes `data/train.parquet`, `data/test.parquet`, `data/split_meta.json` |
| `baselines.py` | Logistic regression / random forest / gradient boosting on the same split → `results/baselines.json` |
| `run_policy_induction.py` | Fit + predict with PolicyInduction → `results/policy_induction.json` |
| `compare.py` | Renders the comparison table |

## Running

```bash
python experiments/compas/prepare_data.py
python experiments/compas/baselines.py
python experiments/compas/run_policy_induction.py --dry-run   # cost estimate, no API calls
python experiments/compas/run_policy_induction.py             # ~3,000 LLM calls
python experiments/compas/compare.py
```

`prepare_data.py` and `baselines.py` are free and take seconds. Only
`run_policy_induction.py` costs money.

## Data preparation decisions

**Row filter.** The raw CSV has 7,214 rows; ProPublica's published filter reduces
it to **6,172**, which is the subset most papers use. `prepare_data.py` applies it
and warns if the count drifts from 6,172 (which would mean the upstream CSV
changed).

**Features (8).** `sex`, `age`, `age_cat`, `juv_fel_count`, `juv_misd_count`,
`juv_other_count`, `priors_count`, `c_charge_degree` — the set used in the
rule-list literature.

**Excluded, deliberately:**

- *Leakage* — `decile_score`, `score_text`, `v_decile_score` are COMPAS's own
  prediction of the very outcome being predicted. `is_recid`, `r_*`,
  `violent_recid`, `start/end/event` are the outcome or only knowable after it.
  Including any of these predicts the answer from the answer.
- *PII* — `name`, `first`, `last`, `dob` are real people's identities and must not
  be sent to an LLM API.
- *`race`* — off by default, available via `--include-race`. CORELS reports that
  no optimal rule list selected a race antecedent, so they excluded it. Running
  both ways is a legitimate experiment in its own right: PolicyInduction emits
  readable policies, so you can see whether race is used, which a black-box model
  would not show you.

`prepare_data.py` asserts no leakage/PII column reaches the feature list, rather
than relying on the list being correct by inspection.

**Split.** Stratified 70/30, `random_state=0` → 4,320 train / 1,852 test, both
45.5% positive. Written to parquet so every model sees byte-identical data.

⚠️ Unlike Adult — which ships an official `adult.data`/`adult.test` split — **COMPAS
has no canonical split**, and the literature does not agree on one. CORELS uses
10-fold CV over ~7,210 rows (train 6,489 / test 721); ProPublica's own analysis
and most fairness papers use the 6,172 subset. Published numbers are therefore a
**sanity reference, not a target**. The trustworthy comparison is the local one,
where every model shares this exact split.

## Baseline results

Test set n=1,000, 44.2% positive.

| Model | Train rows | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | 4,320 | 0.6710 | 0.6050 | 0.6445 | 0.5701 | 0.7173 |
| random_forest | 4,320 | 0.6320 | 0.5701 | 0.5894 | 0.5520 | 0.6569 |
| gradient_boosting | 4,320 | 0.6730 | 0.6237 | 0.6347 | 0.6131 | 0.7204 |
| logistic_regression | 500 | 0.6740 | 0.5571 | 0.6973 | 0.4638 | 0.7088 |
| random_forest | 500 | 0.6160 | 0.5566 | 0.5684 | 0.5452 | 0.6504 |
| gradient_boosting | 500 | 0.6580 | 0.5829 | 0.6323 | 0.5407 | 0.6909 |
| majority class (all NO) | — | 0.5580 | — | — | — | — |

Logistic regression at **67.1% accuracy / AUC 0.717** lands squarely in the
published 65–68% / 0.70–0.73 range, which confirms the pipeline is set up
correctly.

**Two things worth noting before running PolicyInduction:**

1. **This task saturates almost immediately.** LR on 500 rows scores 0.6740 —
   *better* than LR on all 4,320 (0.6710). So PolicyInduction is not handicapped
   by training on 500 rows, and "more data would fix it" is not available as an
   explanation here.
2. **The ceiling is low for everyone.** The best model beats the majority-class
   baseline by ~11 points. Recidivism is genuinely hard to predict from these
   features; a modest PolicyInduction score is expected, not a failure.

Every model is fitted twice — on all 4,320 rows and on the same 500 rows
PolicyInduction uses. The **subsample row is the fair comparison**; comparing a
500-row method against a 4,320-row one would not be meaningful.

## PolicyInduction configuration

Two settings differ from the VCBench runs, both deliberately:

- **`beta=1.0`** instead of the `0.5` default. COMPAS is ~45% positive and the
  literature reports accuracy/F1, so precision and recall should weigh equally.
  VCBench used 0.5 because it is ~9% positive and precision-sensitive.
- **`max_gen_batches=7`**, `policy_batch_size=10`, `max_samples_as_context=20`.

Defaults: 500 train rows, 1,000 test rows, gemini-3.5-flash for generation
(the strongest generator in the VCBench runs) and gemini-2.5-flash-lite for
scoring. All overridable via CLI flags.

Estimated cost at defaults: **~3,000 LLM calls** (7 generation + 1,000 scoring +
up to 2,000 predict). Check with `--dry-run`, which makes zero API calls — it
supplies a stub instructions template, since `set_task()` would otherwise bill a
real call just to write one.

Fit and predict both checkpoint into `run01/`, so re-running the same command
after an interruption resumes rather than restarting.

## Known gaps

- **No ROC-AUC for PolicyInduction.** `predict()` yields a hard YES/NO label, not
  a probability, so AUC is not computable without changing the library. The
  baselines report it for comparison against the literature; the head-to-head
  comparison uses accuracy/F1.
- **Single run.** The VCBench analysis found that two runs of an *identical*
  configuration differed by ~0.08 precision and ~0.15 recall from LLM sampling
  noise alone. One run here is indicative, not conclusive — repeat 3× before
  drawing conclusions.
- **`c_charge_desc` unused.** 389 free-text charge descriptions that traditional
  ML usually one-hot-crushes or drops, but an LLM could read semantically. A
  plausible edge for this method, but it is *extra* information the baselines do
  not get, so it belongs in a separate variant rather than the headline number.

## Version control

`experiments/` is listed in `.gitignore`, so **nothing in this directory is
tracked by git** — including these scripts. That is fine for outputs (`data/`,
`results/`, `run01/`) but means the code here is not backed up or reviewable. If
these scripts should be version-controlled, either move them under `examples/`
(where `.gitignore` un-ignores `*.ipynb`) or add a negation rule for `*.py` under
`experiments/`.
