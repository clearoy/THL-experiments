# ChangeMyView persuasion benchmark

PolicyInduction on the pair task from Tan et al. 2016 (WWW): given two
argumentative replies to the same ChangeMyView post — one that earned a delta,
one that did not — identify the winner.

## Data

`cmv.tar.bz2` → `pair_task/`, v1.0 (January 2016). **3,456 train / 807 heldout
pairs**, both verified against the published counts.

⚠️ The canonical URL `https://chenhaot.com/data/cmv/cmv.tar.bz2` is **dead** —
the site migrated to GitHub Pages and dropped its data directory (the paper page
still links to it). `data/cmv.tar.bz2` here came from the
[2022-12-26 Wayback snapshot](https://web.archive.org/web/20221226140424id_/https://chenhaot.com/data/cmv/cmv.tar.bz2).
The README inside confirms v1.0, not the 11/2016 dump.

### Schema

One JSON object per line: `op_author`, `op_title`, `op_text`, `op_name`,
`positive`, `negative`.

**`positive`/`negative` are dicts, not lists** — the bundled README says "a list
of replies", but they are `{ancestor, author, comments[]}`, where `comments` is
the rooted path-unit as Reddit API objects. Text is `comments[i]["body"]`.

Only `body` is read. Comment metadata is excluded deliberately:
`author_flair_text` carries the author's delta count (`"1∆"`), which would leak
the label.

**71% of positive and 82% of negative units are a single comment**, so
`root_reply` and `full_path` differ on only ~25% of pairs.

## Design

**Fit pointwise, evaluate pairwise.** One row per path-unit (text = OP +
argument, label = delta/no_delta). At evaluation, both members of a heldout pair
are scored independently and the higher one is predicted.

The two members are **never shown side by side** — that formulation carries
~50pp of positional bias on this data and would swamp any real signal.

### Three conditions

| Condition | Text |
|---|---|
| `root_reply` | root comment only — **primary** |
| `full_path` | every comment in the path-unit |
| `root_truncated` | both roots cut to the **shorter one's** word count |

`root_truncated` is the length control. With both sides at identical word
counts, a word-count baseline is forced to exactly 50%. `prepare_data.py` and
`baselines.py` both assert this and **abort** if it drifts — anything else means
the truncation is broken and every downstream number is uninterpretable.

## Baselines (free, no API calls)

Pointwise fit → pairwise eval, same protocol as PolicyInduction.

| Condition | word count | BoW (arg only) | BoW (OP+arg) |
|---|---|---|---|
| `root_reply` | **0.5967** | 0.6146 | 0.6171 |
| `full_path` | 0.6543 | 0.6481 | 0.6667 |
| `root_truncated` | **0.5000** | 0.5874 | 0.5725 |

*(BoW fitted on all 6,912 train units. At `--n-train-units 300`, matching the
LLM budget: 0.5849 / 0.6530 / 0.5725.)*

`root_reply` word count lands on **0.5967**, matching the expected 59.6%.
`root_truncated` word count is **exactly 0.5000**.

BoW on `root_truncated` reaches 0.5874 with length held constant, so there is
real lexical signal beyond length — that is the headroom the rules must beat.

### Targets

| | Accuracy |
|---|---|
| Word count (`root_reply`) | 59.6% |
| Tan et al. 2016 interplay features | 65.1% |
| Labruna et al. 2026, best LLM | 64.53% |

n=807 → **SE ≈ 1.7pp**. Gaps under ~4pp are not real.

## Running

```bash
# 1. free — do these first
python experiments/cmv/prepare_data.py
python experiments/cmv/baselines.py

# 2. cost estimate, zero API calls
python experiments/cmv/run_policy_induction.py --dry-run

# 3. the run: ~1,900 calls
python experiments/cmv/run_policy_induction.py --condition root_reply --seed 0

# 4. the control: one generic prompt, no rules, ~1,600 calls
python experiments/cmv/run_policy_induction.py --control --condition root_reply

# 5. rule set, Tan mapping, accuracy table
python experiments/cmv/analyze.py
```

Full matrix (3 conditions × 3 seeds + 3 controls) is ~22k calls before cache
hits. `--n-eval-pairs 100` cuts a scouting run to ~250.

### Caching

`cache_llm.py` wraps the LLM in a sqlite cache keyed on the exact prompt —
effectively **(rule, text)**, but also separating model and temperature, which a
bare (rule, text) key would collide. Re-runs and overlapping conditions are free.
Cache hits report `total_tokens=0` so spend is not overstated. Disable with
`--no-cache`.

### Models

`--gen-model` (strong, writes policies) defaults to `gemini-3.5-flash`;
`--predict-model` (cheap, evaluates each rule per unit) to
`gemini-2.5-flash-lite`. Rule evaluation is >90% of the calls.

## The control

`--control` scores every unit with **one** generic prompt —

> *"This argument is persuasive enough to change the original poster's mind and
> earn a delta."*

— and no induction. **If the rule set does not beat this by more than ~4pp, the
induction did nothing** and the rules are decorative. `analyze.py` computes the
gap and says so outright.

## Known gaps

- **Rule↔Tan mapping is keyword-based**, so it is a first pass for manual
  review, not a verdict. `analyze.py --verbose` prints matched terms.
- **`op_text` truncated to 200 words** by default (`--op-words 0` for full).
  Some OPs run past 5,000 characters, and it is repeated in every scoring call.
- **542 distinct OPs across 807 heldout pairs** — some OPs appear in multiple
  pairs, so pairs are not fully independent. Train and heldout are separate time
  periods, so there is no train/test leakage.
- **Generation runs at `temperature=1.0`.** Use ≥3 seeds before believing any
  gap.
