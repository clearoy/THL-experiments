"""Prepare the ProPublica COMPAS dataset for the PolicyInduction benchmark.

Downloads the raw CSV (if absent), applies ProPublica's published row filter,
selects the feature set used in the interpretable-rule-list literature, and
writes a fixed stratified train/test split that every model in this experiment
shares.

Run:
    python experiments/compas/prepare_data.py
    python experiments/compas/prepare_data.py --include-race
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_CSV = DATA_DIR / "compas-scores-two-years.csv"

TARGET = "two_year_recid"

# The feature set used by the interpretable-rule-list literature (CORELS et al).
# `race` is appended only with --include-race: CORELS reports that no optimal
# rule list selected a race antecedent, so they excluded it.
BASE_FEATURES = [
    "sex",
    "age",
    "age_cat",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
]

# Never fed to a model. Two distinct reasons, kept separate deliberately.
LEAKAGE = [
    # COMPAS's own prediction of the outcome we are predicting.
    "decile_score",
    "score_text",
    "v_decile_score",
    "v_score_text",
    "type_of_assessment",
    "v_type_of_assessment",
    # The outcome itself, or facts only knowable after it.
    "is_recid",
    "r_case_number",
    "r_charge_degree",
    "r_days_from_arrest",
    "r_offense_date",
    "r_charge_desc",
    "r_jail_in",
    "r_jail_out",
    "violent_recid",
    "is_violent_recid",
    "vr_case_number",
    "vr_charge_degree",
    "vr_offense_date",
    "vr_charge_desc",
    "start",
    "end",
    "event",
]
PII = ["name", "first", "last", "dob"]


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, downloading it on first run."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    print(f"raw: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def propublica_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ProPublica's published row filter, yielding the canonical 6,172.

    Drops rows where the arrest could not be matched to the COMPAS screening
    within +/-30 days, rows with no recidivism record, ordinary traffic
    offences, and rows with no COMPAS score.
    """
    out = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    print(f"after ProPublica filter: {out.shape[0]} rows")
    if out.shape[0] != 6172:
        print(
            f"  WARNING: expected the canonical 6172 rows, got {out.shape[0]}. "
            "The upstream CSV may have changed."
        )
    return out


def build(include_race: bool, test_size: float, seed: int) -> None:
    df = propublica_filter(load_raw())

    features = BASE_FEATURES + (["race"] if include_race else [])
    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from the raw CSV: {missing}")

    # Guard rather than trust: assert nothing excluded slipped into `features`.
    banned = set(LEAKAGE) | set(PII)
    if banned & set(features):
        raise SystemExit(f"Refusing to build: leakage/PII in features {banned & set(features)}")

    data = df[features + [TARGET]].dropna().reset_index(drop=True)
    data["label"] = data[TARGET].map({1: "YES", 0: "NO"})

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=data[TARGET],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_parquet(DATA_DIR / "train.parquet")
    test.to_parquet(DATA_DIR / "test.parquet")

    meta = {
        "source_url": DATA_URL,
        "filter": "ProPublica published filter",
        "rows_after_filter": int(df.shape[0]),
        "rows_after_dropna": int(data.shape[0]),
        "features": features,
        "include_race": include_race,
        "target": TARGET,
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate_train": float(train[TARGET].mean()),
        "positive_rate_test": float(test[TARGET].mean()),
    }
    (DATA_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\ntrain: {len(train)} rows, {train[TARGET].mean():.1%} positive")
    print(f"test:  {len(test)} rows, {test[TARGET].mean():.1%} positive")
    print(f"features ({len(features)}): {features}")
    print(f"\nwrote {DATA_DIR}/train.parquet, test.parquet, split_meta.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--include-race",
        action="store_true",
        help="Include the race column (excluded by default, following CORELS).",
    )
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.include_race, args.test_size, args.seed)


if __name__ == "__main__":
    main()
