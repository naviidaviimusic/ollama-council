"""Adaptive model council for local Ollama models.

Every answer is produced by at least TWO different model families:

    draft (model A) ──┐
                      ├─ agree? ── yes ─▶ answer A                       ("agreed")
    check (model B) ──┘     │
                            no / a member failed
                            ▼
                 model C joins → JSON: majority vote                   ("escalated-vote")
                               → prose: synthesizer merges the answers ("escalated-synthesis")

Agreement is judged by meaning, not word overlap: embedding cosine
(nomic-embed-text) for prose, and for JSON the share of short categorical fields
(risk level, label, yes/no) that match. Word overlap scored two "risk: high"
answers as 0.18 agreement. Falls back to word overlap if no embed model.

strategy="full" asks every member every time and synthesizes (slower, ~2x calls).
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import client, guard, queue

DEFAULT_MEMBERS = ["gemma2:2b", "llama3.2", "qwen2.5:3b"]
CACHE_DB = Path.home() / ".cache" / "ollama-council" / "cache.db"

SYNTH_PROMPT = (
    "Several independent analysts answered the same question{why}. Write ONE final answer: "
    "keep what they agree on, resolve disagreements with the strongest reasoning, drop "
    "anything unsupported. Answer the question directly; do not mention the analysts."
    "\n\nQUESTION:\n{q}\n\nANSWERS:\n{a}")


@dataclass
class Result:
    answer: str
    # agreed | escalated-vote | escalated-synthesis | full-vote | full-synthesis | cache | single | failed
    mode: str
    agreement: float
    members: dict = field(default_factory=dict)   # model -> its own answer
    secs: float = 0.0
    flagged_injection: bool = False
    redacted: int = 0

    def __str__(self):
        return self.answer


def _overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(1, len(wa | wb))


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def json_fields(text: str) -> dict:
    d = guard.salvage_json(text)
    return {k: str(v).strip().lower() for k, v in d.items()
            if isinstance(v, (str, int, float, bool))}


class Council:
    def __init__(self, members=None, synthesizer=None, strategy="adaptive",
                 agree_prose=0.80, agree_json=0.5, cache_hours=6.0,
                 timeout=120, embed_model="nomic-embed-text", use_queue=True,
                 generate=None, embed=None):
        self.members = list(members or DEFAULT_MEMBERS)
        if not self.members:
            raise ValueError("need at least one member model")
        self.synthesizer = synthesizer or self.members[0]
        self.strategy = strategy
        self.agree_prose = agree_prose
        self.agree_json = agree_json
        self.cache_hours = cache_hours
        self.timeout = timeout
        self.embed_model = embed_model
        self.use_queue = use_queue
        self._generate = generate or client.generate     # injectable for tests
        self._embed_fn = embed or (lambda t: client.embed(t, self.embed_model))

    # ── one model ──────────────────────────────────────────────────────────
    def _one(self, model, prompt, system, max_tokens, temperature, fmt):
        try:
            return self._generate(prompt, model=model, system=system, max_tokens=max_tokens,
                                  temperature=temperature, fmt=fmt, timeout=self.timeout)
        except client.OllamaTimeout as e:
            return f"[TIMEOUT] {e}"
        except Exception as e:
            return f"[ERROR] {e}"

    def agreement(self, a: str, b: str, wants_json: bool, emb: dict) -> float:
        if wants_json:
            fa, fb = json_fields(a), json_fields(b)
            keys = [k for k in fa if k in fb and len(fa[k]) <= 20 and len(fb[k]) <= 20]
            if keys:
                return sum(fa[k] == fb[k] for k in keys) / len(keys)
        if a not in emb:
            emb[a] = self._embed_fn(a)
        if b not in emb:
            emb[b] = self._embed_fn(b)
        if emb[a] and emb[b]:
            return _cosine(emb[a], emb[b])
        return _overlap(a, b)

    # ── cache ──────────────────────────────────────────────────────────────
    def _cache(self, key, value=None):
        if not self.cache_hours:
            return None
        try:
            CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(CACHE_DB), timeout=30)
            con.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
            if value is None:
                row = con.execute("SELECT v FROM c WHERE k=? AND ts>?",
                                  (key, time.time() - self.cache_hours * 3600)).fetchone()
                con.close()
                return row[0] if row else None
            con.execute("INSERT OR REPLACE INTO c VALUES (?,?,?)", (key, value, time.time()))
            con.execute("DELETE FROM c WHERE ts<?", (time.time() - 7 * 86400,))
            con.commit()
            con.close()
        except sqlite3.Error:
            return None

    def _synthesize(self, prompt, answers, system, max_tokens, why=""):
        joined = "\n\n".join(f"[{m}]\n{a}" for m, a in answers.items())
        return self._one(self.synthesizer,
                         SYNTH_PROMPT.format(why=why, q=prompt[:3000], a=joined[:6000]),
                         system, max_tokens, 0.3, None)

    # ── public ─────────────────────────────────────────────────────────────
    def ask(self, prompt: str, system: str = None, max_tokens: int = 800,
            temperature: float = 0.7, schema=None) -> Result:
        """schema: "json" or a JSON-schema dict → Ollama constrains decoding to it."""
        t0 = time.time()
        prompt, flagged = guard.screen(prompt)
        wants_json = bool(schema) or "json" in f"{system or ''} {prompt}".lower()
        key = hashlib.sha256(json.dumps(
            [self.members, self.strategy, system, prompt, max_tokens, schema],
            sort_keys=True, default=str).encode()).hexdigest()
        hit = self._cache(key)
        if hit:
            return Result(hit, "cache", 1.0, flagged_injection=flagged)

        if self.use_queue:
            with queue.slot() as got:
                if not got:
                    return Result("[TIMEOUT] model queue busy", "failed", 0.0)
                res = self._decide(prompt, system, max_tokens, temperature, schema, wants_json)
        else:
            res = self._decide(prompt, system, max_tokens, temperature, schema, wants_json)

        # second layer: verify, then redact secrets from whatever we return
        if not guard.valid(res.answer, wants_json):
            ok = [a for a in res.members.values() if guard.valid(a, wants_json)]
            if ok:
                res.answer = ok[0]
        res.answer, res.redacted = guard.redact(res.answer)
        res.flagged_injection = flagged
        res.secs = round(time.time() - t0, 1)
        if guard.valid(res.answer, wants_json):
            self._cache(key, res.answer)
        return res

    def _decide(self, prompt, system, max_tokens, temperature, schema, wants_json) -> Result:
        answers, last = {}, ""

        def get(m):
            nonlocal last
            r = self._one(m, prompt, system, max_tokens, temperature, schema)
            last = r
            if r and not r.startswith(guard.FAILED):
                answers[m] = r

        emb = {}
        if self.strategy == "full":
            for m in self.members:
                get(m)
            if not answers:
                return Result(last, "failed", 0.0)
            return self._resolve(prompt, system, max_tokens, answers, wants_json, emb, "full")

        get(self.members[0])                       # draft
        if len(self.members) > 1:
            get(self.members[1])                   # independent cross-check
        if len(answers) == 2:
            a, b = answers.values()
            score = self.agreement(a, b, wants_json, emb)
            if score >= (self.agree_json if wants_json else self.agree_prose):
                return Result(a, "agreed", round(score, 3), dict(answers))
        for m in self.members[2:]:                 # disagreement or a failed member: escalate
            get(m)
        if not answers:
            return Result(last, "failed", 0.0)
        if len(answers) == 1:
            return Result(next(iter(answers.values())), "single", 0.0, dict(answers))
        return self._resolve(prompt, system, max_tokens, answers, wants_json, emb, "escalated")

    def _resolve(self, prompt, system, max_tokens, answers, wants_json, emb, prefix) -> Result:
        names = list(answers)
        pair = {m: sum(self.agreement(answers[m], answers[n], wants_json, emb)
                       for n in names if n != m) / max(1, len(names) - 1) for m in names}
        best = max(pair, key=pair.get)
        score = round(sum(pair.values()) / len(pair), 3)
        ranked = dict(sorted(answers.items(), key=lambda kv: -pair[kv[0]]))
        if wants_json or len(names) < 2:
            return Result(answers[best], f"{prefix}-vote", score, ranked)
        synth = self._synthesize(prompt, answers, system, max_tokens,
                                 why=" and DISAGREED" if prefix == "escalated" else "")
        if synth and not synth.startswith(guard.FAILED):
            return Result(synth, f"{prefix}-synthesis", score, ranked)
        return Result(answers[best], f"{prefix}-vote", score, ranked)


def ask(prompt: str, **kwargs) -> Result:
    """One-shot helper: ask(prompt, members=[...], system=..., schema="json")."""
    init_keys = {"members", "synthesizer", "strategy", "agree_prose", "agree_json",
                 "cache_hours", "timeout", "embed_model", "use_queue"}
    council = Council(**{k: kwargs.pop(k) for k in list(kwargs) if k in init_keys})
    return council.ask(prompt, **kwargs)
