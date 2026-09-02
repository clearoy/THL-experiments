"""Disk-cached LLM wrapper, keyed on the exact prompt.

PolicyInduction scores every (policy, sample) pair, so the same rule is asked
about the same text many times across seeds, conditions, and re-runs. This
wrapper makes those repeats free.

The key is a hash of (instructions, query, response_format, models,
temperature). Since the policy text and the unit text both live inside `query`,
that is effectively a cache on (rule, text) -- while still separating calls that
differ by model or temperature, which a bare (rule, text) key would collide.

Injected via PolicyInduction(..., _llm=CachingLLM()).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Type

from think_reason_learn.core.llms import llm as real_llm
from think_reason_learn.core.llms._schemas import NOT_GIVEN, LLMResponse, NotGiven

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "data" / "llm_cache.sqlite"


class CachingLLM:
    """Drop-in for the global `llm`, backed by a sqlite cache."""

    def __init__(self, db_path: Path | str = DEFAULT_DB, enabled: bool = True) -> None:
        self.db_path = Path(db_path)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  k TEXT PRIMARY KEY, payload TEXT, provider TEXT,"
            "  model TEXT, total_tokens INTEGER)"
        )
        self._conn.commit()

    # ── key ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _models(llm_priority: List[Any]) -> str:
        out = []
        for c in llm_priority:
            out.append(c["model"] if isinstance(c, dict) else getattr(c, "model", str(c)))
        return ",".join(out)

    def _key(self, query, llm_priority, response_format, instructions, temperature) -> str:
        fmt = getattr(response_format, "__name__", str(response_format))
        instr = "" if isinstance(instructions, NotGiven) or instructions is None else instructions
        temp = "" if isinstance(temperature, NotGiven) or temperature is None else str(temperature)
        blob = "\x00".join([instr, query, fmt, self._models(llm_priority), temp])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ── (de)serialisation ──────────────────────────────────────────────────────

    @staticmethod
    def _dump(resp: Any) -> str:
        if resp is None:
            return json.dumps({"_none": True})
        if isinstance(resp, str):
            return json.dumps({"_str": resp})
        return json.dumps({"_model": resp.model_dump()})

    @staticmethod
    def _load(payload: str, response_format: Type[Any]) -> Any:
        d = json.loads(payload)
        if "_none" in d:
            return None
        if "_str" in d:
            return d["_str"]
        return response_format(**d["_model"])

    # ── the interface PolicyInduction calls ────────────────────────────────────

    async def respond(
        self,
        query: str,
        llm_priority: List[Any],
        response_format: Type[Any],
        instructions: str | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        **kwargs: Dict[str, Any],
    ) -> LLMResponse[Any]:
        if not self.enabled:
            return await real_llm.respond(
                query=query, llm_priority=llm_priority, response_format=response_format,
                instructions=instructions, temperature=temperature, **kwargs,
            )

        k = self._key(query, llm_priority, response_format, instructions, temperature)
        row = self._conn.execute(
            "SELECT payload, provider, model, total_tokens FROM cache WHERE k=?", (k,)
        ).fetchone()
        if row is not None:
            self.hits += 1
            payload, provider, model, tokens = row

            # total_tokens is reported as 0 on a hit: the call cost nothing this
            # time, and counting cached tokens would overstate real spend.
            return LLMResponse(
                response=self._load(payload, response_format),
                logprobs=[],
                total_tokens=0,
                provider_model=_fake_choice(provider, model),
            )

        self.misses += 1
        resp = await real_llm.respond(
            query=query, llm_priority=llm_priority, response_format=response_format,
            instructions=instructions, temperature=temperature, **kwargs,
        )
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (
                    k,
                    self._dump(resp.response),
                    resp.provider_model.provider,
                    resp.provider_model.model,
                    int(resp.total_tokens or 0),
                ),
            )
            self._conn.commit()
        return resp

    # ── reporting ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        n = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"hits": self.hits, "misses": self.misses, "rows_in_cache": int(n)}

    def close(self) -> None:
        self._conn.close()


def _fake_choice(provider: str, model: str):
    """Rebuild the provider_model stamp for a cached response."""
    from think_reason_learn.core.llms import (
        AnthropicChoice, GoogleChoice, OpenAIChoice, XAIChoice,
    )

    cls = {
        "anthropic": AnthropicChoice, "google": GoogleChoice,
        "openai": OpenAIChoice, "xai": XAIChoice,
    }.get(provider, OpenAIChoice)
    return cls(model=model)
