"""Pure-embedding baseline on the CMV pair task.

A ceiling check for PolicyInduction. Same 807 heldout pairs, same pairwise
metric, no rules and no LLM judgement -- just dense vectors and a linear model.
If embeddings clear the word-count baseline, the signal is reachable by cheap
means and the induced-rule pipeline is failing at something other than the task
being hard.

The features are DIFFERENCES between the two replies, and the classifier is
fitted with no intercept. That makes the model exactly antisymmetric: swapping
argument_A and argument_B negates every feature and flips the prediction. So
positional bias is impossible by construction, and augmenting with both orders
would add nothing -- the loss is already symmetric.

One consequence worth stating: anything depending only on the post cancels in
the difference, because both replies share it. The post can therefore only
enter through an interaction, which is what the cosine feature is -- the dense
analogue of the word-overlap "interplay" feature Tan et al. found predictive.

Embeddings are cached to data/embeddings_<model>.npz keyed by text hash, so
re-runs cost nothing.

Run:
    python experiments/cmv/embedding_baseline.py
    python experiments/cmv/embedding_baseline.py --condition root_truncated
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]

_ENV = REPO_ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from google import genai  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

CONDITIONS = ("root_reply", "full_path", "root_truncated")
CS = (0.001, 0.01, 0.1, 1.0, 10.0)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class EmbeddingStore:
    """Disk-cached embeddings, keyed on a hash of the exact text."""

    def __init__(self, model: str, batch: int, concurrency: int) -> None:
        self.model = model
        self.batch = batch
        self.path = DATA_DIR / f"embeddings_{model.replace('/', '_')}.npz"
        self.sem = asyncio.Semaphore(concurrency)
        self.client = genai.Client(
            api_key=os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.cache: Dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.cache = {k: z[k] for k in z.files}
        self.hits = self.misses = 0

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.aio.models.embed_content(
                        model=self.model, contents=texts
                    )
                    return [np.asarray(e.values, dtype=np.float32) for e in r.embeddings]
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = sorted({t for t in texts if _key(t) not in self.cache})
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            chunks = [todo[i : i + self.batch] for i in range(0, len(todo), self.batch)]
            print(f"  embedding {len(todo)} new texts in {len(chunks)} batches...")
            done = 0
            for group in [chunks[i : i + 8] for i in range(0, len(chunks), 8)]:
                out = await asyncio.gather(*(self._embed_batch(c) for c in group))
                for c, vecs in zip(group, out):
                    for t, v in zip(c, vecs):
                        self.cache[_key(t)] = v
                done += sum(len(c) for c in group)
                print(f"    {done}/{len(todo)}", end="\r", flush=True)
            print()
            np.savez_compressed(self.path, **self.cache)
        return np.stack([self.cache[_key(t)] for t in texts])


def unit_norm(M: np.ndarray) -> np.ndarray:
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)


def build_pairs(df: pd.DataFrame, condition: str, op_words: int, seed: int):
    """One row per pair with A/B randomised, mirroring run_policy_induction."""
    wide = df.pivot(index="pair_id", columns="side", values=f"arg_{condition}").dropna()
    wc = df.pivot(index="pair_id", columns="side", values=f"words_{condition}").reindex(wide.index)
    op = df.groupby("pair_id")[["op_title", "op_text"]].first().reindex(wide.index)
    op_text = (op["op_title"] + "\n\n" + op["op_text"]).str.split().str[:op_words].str.join(" ")

    n = len(wide)
    a_is_pos = np.zeros(n, dtype=bool)
    a_is_pos[: n // 2] = True
    np.random.default_rng(seed).shuffle(a_is_pos)

    pos, neg = wide["positive"].to_numpy(), wide["negative"].to_numpy()
    wpos, wneg = wc["positive"].to_numpy(float), wc["negative"].to_numpy(float)
    return pd.DataFrame(
        {
            "op": op_text.to_numpy(),
            "arg_A": np.where(a_is_pos, pos, neg),
            "arg_B": np.where(a_is_pos, neg, pos),
            "words_A": np.where(a_is_pos, wpos, wneg),
            "words_B": np.where(a_is_pos, wneg, wpos),
            "y": a_is_pos.astype(int),
        },
        index=wide.index,
    )


async def featurise(frame: pd.DataFrame, store: EmbeddingStore) -> Dict[str, np.ndarray]:
    """Antisymmetric feature blocks. Every one negates when A and B swap."""
    E_a = unit_norm(await store.embed(frame["arg_A"].tolist()))
    E_b = unit_norm(await store.embed(frame["arg_B"].tolist()))
    E_op = unit_norm(await store.embed(frame["op"].tolist()))

    diff = E_a - E_b
    interplay = ((E_op * E_a).sum(1) - (E_op * E_b).sum(1)).reshape(-1, 1)
    length = (np.log1p(frame["words_A"].to_numpy())
              - np.log1p(frame["words_B"].to_numpy())).reshape(-1, 1)
    return {
        "embedding difference": diff,
        "OP-similarity difference": interplay,
        "log-length difference": length,
        "embedding + OP-similarity": np.hstack([diff, interplay]),
        "embedding + OP-sim + length": np.hstack([diff, interplay, length]),
    }


def fit_eval(Xtr, ytr, Xte, yte, seed: int) -> Dict[str, float]:
    """No intercept: keeps the model exactly antisymmetric. C by 5-fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_c, best_cv = CS[0], -np.inf
    for c in CS:
        accs = []
        for tr, va in skf.split(Xtr, ytr):
            lr = LogisticRegression(C=c, fit_intercept=False, max_iter=3000)
            lr.fit(Xtr[tr], ytr[tr])
            accs.append((lr.predict(Xtr[va]) == ytr[va]).mean())
        if np.mean(accs) > best_cv:
            best_c, best_cv = c, float(np.mean(accs))
    lr = LogisticRegression(C=best_c, fit_intercept=False, max_iter=3000)
    lr.fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    acc = float(np.where(p == 0.5, 0.5, (p > 0.5) == yte).mean())
    return {"accuracy": acc, "cv_accuracy": best_cv, "C": best_c}


async def main_async(args: argparse.Namespace) -> None:
    train = pd.read_parquet(DATA_DIR / "units_train.parquet")
    heldout = pd.read_parquet(DATA_DIR / "units_heldout.parquet")
    tr = build_pairs(train, args.condition, args.op_words, args.seed)
    te = build_pairs(heldout, args.condition, args.op_words, args.seed)

    store = EmbeddingStore(args.model, args.batch, args.concurrency)
    print(f"condition : {args.condition}   model: {args.model}")
    print(f"train     : {len(tr)} pairs      heldout: {len(te)} pairs")
    Ftr = await featurise(tr, store)
    Fte = await featurise(te, store)
    print(f"cache     : {store.hits} hits, {store.misses} embedded\n")

    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    sizes = [n for n in (args.n_train_pairs, 0) if n < len(tr)] + [len(tr)]
    sizes = sorted(set(sizes) - {0})

    rng = np.random.default_rng(args.seed)
    order = np.concatenate([rng.permutation(np.where(ytr == 1)[0]),
                            rng.permutation(np.where(ytr == 0)[0])])

    rows = []
    hdr = f"{'features':30s} " + " ".join(f"{'n=' + str(n):>12s}" for n in sizes)
    print(hdr); print("-" * len(hdr))
    for name in Ftr:
        cells = []
        for n in sizes:
            half = n // 2
            idx = np.concatenate([order[:half], order[len(order) // 2 : len(order) // 2 + n - half]])
            r = fit_eval(Ftr[name][idx], ytr[idx], Fte[name], yte, args.seed)
            rows.append({"features": name, "n_train_pairs": int(n), **r})
            cells.append(f"{r['accuracy']:12.4f}")
        print(f"{name:30s} " + " ".join(cells))

    print(f"\n{'reference':30s}")
    for k, v in (("word count (this repo)", 0.5967), ("Tan et al. 2016", 0.6510),
                 ("PolicyInduction, 15 rules", 0.4963), ("control, one prompt", 0.5509)):
        print(f"  {v:.4f}  {k}")
    print(f"\n  n={len(te)} pairs -> SE ~1.8pp")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"embeddings_{args.condition}_seed{args.seed}.json"
    out.write_text(json.dumps(
        {"condition": args.condition, "seed": args.seed, "model": args.model,
         "op_words": args.op_words, "n_heldout_pairs": int(len(te)), "results": rows},
        indent=2))
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, default="root_reply")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="gemini-embedding-001")
    p.add_argument("--n-train-pairs", type=int, default=500,
                   help="Extra ladder point matching the LLM budget.")
    p.add_argument("--op-words", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=4)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
