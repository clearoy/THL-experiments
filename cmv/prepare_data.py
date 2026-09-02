"""Build pointwise unit tables from the Tan et al. 2016 CMV pair_task.

Each pair contributes TWO rows: the positive path-unit (won a delta) and the
negative one (did not). Fitting is pointwise on these rows; evaluation is
pairwise, by scoring both members of a pair and taking the higher.

Three text conditions are precomputed per unit so every downstream script sees
byte-identical inputs:

  root_reply      the root comment only (primary condition)
  full_path       every comment in the rooted path-unit, concatenated
  root_truncated  both roots cut to the SHORTER one's word count

root_truncated exists to prove the induced rules are not a length proxy: with
both sides at identical word counts, a word-count baseline must score exactly
50%. Anything else means the truncation is broken.

Run:
    python experiments/cmv/prepare_data.py
"""

from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PAIR_DIR = DATA_DIR / "pair_task"

SPLITS = {"train": "train_pair_data.jsonlist.bz2", "heldout": "heldout_pair_data.jsonlist.bz2"}
EXPECTED = {"train": 3456, "heldout": 807}

CONDITIONS = ("root_reply", "full_path", "root_truncated")


def read_pairs(path: Path) -> Iterator[Dict[str, Any]]:
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def unit_text(unit: Dict[str, Any], root_only: bool) -> str:
    """Concatenate the path-unit's comment bodies.

    Only `body` is used. Comment metadata is deliberately excluded --
    `author_flair_text` carries the author's delta count ("1∆"), which would
    leak the label.
    """
    comments = unit["comments"]
    if root_only:
        comments = comments[:1]
    return "\n\n".join(c["body"] for c in comments).strip()


def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def build_split(name: str, drop_deleted: bool) -> pd.DataFrame:
    path = PAIR_DIR / SPLITS[name]
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Extract the archive first:\n"
            f"  tar -xjf {DATA_DIR}/cmv.tar.bz2 -C {DATA_DIR} README pair_task/"
        )

    pairs = list(read_pairs(path))
    print(f"{name}: {len(pairs)} pairs", end="")
    if len(pairs) != EXPECTED[name]:
        print(f"  WARNING: expected {EXPECTED[name]}")
    else:
        print("  (matches the published count)")

    rows: List[Dict[str, Any]] = []
    dropped = 0
    for i, p in enumerate(pairs):
        pos_root = unit_text(p["positive"], root_only=True)
        neg_root = unit_text(p["negative"], root_only=True)

        if drop_deleted and _is_deleted(pos_root, neg_root):
            dropped += 1
            continue

        # Both roots cut to the shorter one's length, so the pair is exactly
        # matched on word count and a length baseline is forced to 50%.
        keep = min(len(pos_root.split()), len(neg_root.split()))

        for side, unit, root in (
            ("positive", p["positive"], pos_root),
            ("negative", p["negative"], neg_root),
        ):
            full = unit_text(unit, root_only=False)
            rows.append(
                {
                    "pair_id": i,
                    "side": side,
                    "label": "YES" if side == "positive" else "NO",
                    "delta": 1 if side == "positive" else 0,
                    "op_name": p["op_name"],
                    "op_title": p["op_title"],
                    "op_text": p["op_text"],
                    "arg_root_reply": root,
                    "arg_full_path": full,
                    "arg_root_truncated": truncate_words(root, keep),
                    "n_comments": len(unit["comments"]),
                }
            )

    df = pd.DataFrame(rows)
    for c in CONDITIONS:
        df[f"words_{c}"] = df[f"arg_{c}"].str.split().str.len()
    if dropped:
        print(f"  dropped {dropped} pairs with a [deleted]/[removed] root")
    return df


def _is_deleted(*texts: str) -> bool:
    return any(t.strip() in ("[deleted]", "[removed]", "") for t in texts)


def length_baseline(df: pd.DataFrame, condition: str) -> float:
    """'Longer argument wins' accuracy. Ties count as 0.5 (random tie-break)."""
    w = df.pivot(index="pair_id", columns="side", values=f"words_{condition}")
    wins = (w["positive"] > w["negative"]).sum()
    ties = (w["positive"] == w["negative"]).sum()
    return float((wins + 0.5 * ties) / len(w))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--keep-deleted",
        action="store_true",
        help="Keep pairs whose root comment is [deleted]/[removed] (dropped by default).",
    )
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        df = build_split(name, drop_deleted=not args.keep_deleted)
        out = DATA_DIR / f"units_{name}.parquet"
        df.to_parquet(out)
        n_pairs = df["pair_id"].nunique()
        print(f"  -> {len(df)} units ({n_pairs} pairs) -> {out.name}")
        print("     word-count baseline:")
        for c in CONDITIONS:
            acc = length_baseline(df, c)
            flag = ""
            if c == "root_truncated":
                flag = "  <- must be 0.500" + ("  OK" if abs(acc - 0.5) < 1e-9 else "  BROKEN")
            print(f"       {c:16s} {acc:.4f}{flag}")
        print()

    meta = {
        "source": "Tan et al. 2016, WWW. cmv.tar.bz2 pair_task (v1.0, Jan 2016)",
        "conditions": list(CONDITIONS),
        "dropped_deleted": not args.keep_deleted,
    }
    (DATA_DIR / "prep_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
