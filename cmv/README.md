# ChangeMyView persuasion benchmark

Pair task from Tan et al. 2016 (WWW). Two replies to the same post, one earned a
delta, one did not. Pick the winner. Their approach is summarised in
[`tan_paper/`](tan_paper/).

`prepare_data.py` → `data/units_{train,heldout}.parquet`, two rows per pair.

| Split | Period | Pairs | Rows |
|---|---|---|---|
| train | 2013-01-01 → 2015-05-07 | 3,456 | 6,912 |
| heldout | 2015-05-08 → 2015-09-01 | 807 | 1,614 |

All methods are evaluated on the same heldout pairs — 807, or 806 for the LLM
run, which fails to score one of them. Accuracy is pairwise,
ties count 0.5. SE on a single accuracy is ≈1.8pp; comparisons use the paired SE
in the Significance table.

## Input

One row per **pair**: the post plus both replies, labelled with the winner.
Which reply lands in slot A is randomised 50/50, so position carries no
information.

| Column | Used for |
|---|---|
| `pair_id` | identifies the pair; one row per pair after pivoting |
| `side` | `positive` / `negative` — which reply won, i.e. the target |
| `label` / `delta` | `YES`/`NO` and `1`/`0`, same target restated |
| `words_<condition>` | word-count baseline |
| `arg_<condition>` | becomes `argument_A` and `argument_B` |
| `op_title`, `op_text` | `op_text` cut to 300 words, renamed `op_view` |

| `<condition>` | Argument text |
|---|---|
| `root_reply` | root comment only — **primary** |
| `full_path` | all comments in the path-unit |
| `root_truncated` | root cut to the shorter side's word count |

PolicyInduction receives four fields — `op_title`, `op_view`, `argument_A`,
`argument_B` — and a YES/NO label meaning "A won".

## Example row

Heldout, `pair_id=196`, `side=positive`.

```
pair_id              : 196
side                 : positive
label / delta        : YES / 1
op_title             : CMV: guys and girls can NEVER be 'just friends'
op_text              : First Let me just say I am 16 and don't have much
                       experience with how it is in the 'adult world'. But I
                       don't think a 100% platonic relationship can exist
                       between a straight male and a straight female. [...]
arg_root_reply       : This sort of logic seems a bit backwards to me.  To me, a
                       relationship is not possible without the people being
                       friends first.  To say the possibility of a relationship
                       precludes a true friendship doesn't make any sense to me,
                       because there is no possibility of a relationship without
                       the friendship.

                       To think about it from a different perspective, do you
                       think that bi people are incapable of having a friendship?
words_root_reply     : 71
words_root_truncated : 71
n_comments           : 1
```

Its partner row is `side=negative`, same `pair_id` and same `op_*`, with
`words_root_reply=94`.

## Experiments

| Name | Definition |
|---|---|
| **word count** | Longer reply wins. Nothing fitted. |
| **BoW** | TF-IDF (1–2 grams) + logistic regression, fitted pointwise on `delta`, evaluated pairwise. Kept in `baselines.json`, off the headline table. |
| **embeddings** | `gemini-embedding-001` (3072d) on each reply. Feature is the **difference** of the two vectors, optionally plus cos(post, reply) difference and log-length difference. Logistic regression with **no intercept**, so the model is exactly antisymmetric and positional bias is impossible. |
| **PolicyInduction** | 10 rounds generate ≤15 comparative rules from 20 labelled pairs each; a scorer answers every rule YES/NO per pair; L1 logistic regression on those bits. Every prompt is the library default except the task description. |
| **control** | Identical pipeline with induction removed — one hand-written prompt ("Argument A is more persuasive…") replaces the induced rules. Isolates whether induction contributes anything. *No control result is currently on disk; see Removed runs.* |
| **Tan et al. 2016** | Quoted from the paper, not run here. Their classifier compares the two replies as a pair and trains on all 3,456. |

Every LLM run scores each heldout pair in **both orders** and averages, which
cancels positional bias. Pairs where both orders name the same slot score 0.5.

## Results

`root_reply`, seed 0, 807 heldout pairs (806 scored by the LLM run — pair 707
fails to score and is absent from it).

| Method | Train pairs | Scorer | Accuracy |
|---|---|---|---|
| Tan et al. 2016 | 3,456 | — | 0.6510 |
| embeddings + cos + length | 3,456 | — | 0.6481 |
| embeddings only | 3,456 | — | 0.6394 |
| embeddings + cos + length | 500 | — | 0.6344 |
| **PolicyInduction, 15 rules** | **500** | **gemini-3.5-flash** | **0.6241** |
| word count | — | — | 0.5967 |

Word count by condition: 0.5967 / 0.6543 / 0.5000.
BoW argument-only by condition: 0.5849 / 0.6530 / 0.5725 — still in
`baselines.json`, kept off the headline table because it is fitted on only 150
pairs.

### Significance

Comparisons use a **paired** standard error, computed per pair, since the methods
are scored on the same pairs. The ±1.8pp figure is the error on a *single*
accuracy and should not be used to compare rows.

| Comparison | Difference | Paired SE | z |
|---|---|---|---|
| rules vs word count | +0.0279 | 0.0217 | 1.29 |
| rules vs embeddings at 500 | −0.0149 | 0.0200 | −0.74 |

PolicyInduction is **not** separated from counting words, and is statistically
level with a dense-embedding baseline at the same training budget.

### Notes

- **cos(post, reply) carries nothing** (0.4919 / 0.5087). The dense analogue of
  Tan's word-overlap feature does not reproduce it, because the real signal is
  content-word divergence *plus* stopword similarity, and a single cosine cancels
  the two.
- **Embeddings are not just length.** Correlation with log-length difference is
  0.635, but on the quartile of pairs closest in length, word count falls to
  0.5074 while embeddings hold 0.5990.
- **The 15 rules are highly redundant.** Each scores 0.604–0.622 alone; together
  0.6241, and refitting weights on heldout itself reaches only 0.6303.
- **Training fit is not recorded** by the pipeline, only the cross-validated
  F-beta. Computed from the saved artifacts, the flash run scores 0.7020 accuracy
  and 0.7118 AUC on its 500 training rows, against 0.6241 / 0.6436 on heldout.
- **Cross-validated accuracy runs ~3pp optimistic**, because the length signal is
  stronger in the training period (word count 0.6306 train vs 0.5967 heldout).
  Quote heldout numbers only.
- **A and B are rarely the same length** on `root_reply`: 1 pair in 807. The median
  pair has one reply 1.73× the other. Only `root_truncated` closes that channel,
  and PolicyInduction has never been run on it.
- **Non-LLM baselines save aggregates only.** `baselines.json` and
  `embeddings_*.json` hold no per-pair predictions, so the paired tests above
  require refitting them. Free — no API calls — but not readable straight from
  disk.

### Removed runs

Earlier runs on `gemini-2.5-flash-lite`, and both no-induction controls, have
been deleted from `results/` and `runs/`. Two findings came from them and are
recorded here because they shaped the current configuration, but they are **no
longer reproducible from files in this repo**:

- Swapping the scorer from `gemini-2.5-flash-lite` to `gemini-3.5-flash`, holding
  the pipeline and the 500 training pairs fixed, moved accuracy 0.4963 → 0.6241.
  Rule fire rates went from 9 of 15 below 5% to all 15 in the 51–57% range. That
  run also regenerated its rules, so it was not a single-variable ablation.
- The control — the same pipeline with one hand-written prompt replacing the
  induced rules — scored 0.6036 on `gemini-3.5-flash`. Against the rules on a
  matched scorer that is +2.1pp at z = 1.71, i.e. induction was **not** shown to
  beat simply asking. Rerunning it costs ~1,714 calls.

## Run

```bash
tar -xjf experiments/cmv/data/cmv.tar.bz2 -C experiments/cmv/data README pair_task/
```

```bash
python experiments/cmv/prepare_data.py
python experiments/cmv/baselines.py
python experiments/cmv/embedding_baseline.py
python experiments/cmv/run_policy_induction.py --condition root_reply --seed 0 --predict-model gemini-3.5-flash
python experiments/cmv/run_policy_induction.py --control --condition root_reply --n-train-pairs 100 --predict-model gemini-3.5-flash
python experiments/cmv/analyze.py
```

`--dry-run` estimates cost with zero calls. Rules run ≈ 2,125 calls (11
generation, 500 fit scoring, 1,614 predict); control ≈ 1,714; embeddings ≈
11,600, cached to `data/embeddings_*.npz`. LLM responses are sqlite-cached on
the exact prompt including the model name.

Generation runs at `temperature=1.0` and `random_state` seeds sampling only, so
re-running regenerates different rules. Use ≥3 seeds before believing any gap.

Data source: the canonical URL is dead; `data/cmv.tar.bz2` came from the
[2022-12-26 Wayback snapshot](https://web.archive.org/web/20221226140424id_/https://chenhaot.com/data/cmv/cmv.tar.bz2),
v1.0.
