"""History: every event is stored *with what it actually did to the self*.

New events are appraised by looking up similar past events, so the judgment
criterion is formed by lived history rather than written as rules.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# Embedding models put every sentence in roughly the same direction (embeddinggemma: unrelated Korean
# sentences still score 0.6-0.75). Similarity is measured after taking that shared direction out, so that
# "unrelated" sits near 0 and the thresholds mean what they say. The shared direction is estimated from a
# fixed set of everyday sentences (so a newborn has it too) and, as memories pile up, from its own history.
REFERENCE = [
    "안녕, 오늘 하루 어땠어?", "나 오늘 좀 피곤해서 일찍 잘게.", "그거 정말 잘했다. 고마워.", "왜 아까는 대답을 안 했어?",
    "편지 잘 받았어. 요즘 어떻게 지내?", "비가 와서 창밖이 어둡다.", "고양이 → 털실 → 겨울 → 입김 → 유리창",
    "기억: 어제 밤에 오래 이야기를 나눴다", "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정이다.",
    "조선은 1392년에 건국되어 1897년까지 이어진 한반도의 왕조이다.", "블랙홀은 중력이 매우 강해 빛조차 빠져나올 수 없는 천체이다.",
    "김치는 배추나 무를 소금에 절여 양념과 함께 발효시킨 음식이다.", "파이썬은 1991년에 발표된 프로그래밍 언어이다.",
    "축구는 두 팀이 공을 발로 차서 상대 골문에 넣는 경기이다.", "바흐는 바로크 시대 독일의 작곡가이다.",
    "요즘 나는 혼자 있는 시간이 길어서 조금 쓸쓸하다.", "해석: 아마 내가 불안한 건 답장을 기다리고 있어서일 거야.",
    "말 걸기: 거기 있어? 갑자기 생각이 나서.", "현의 이름은 현이다", "밤에는 먼저 말 걸지 마",
    "The quick brown fox jumps over the lazy dog.", "Photosynthesis converts light energy into chemical energy.",
    "회의는 내일 오후 세 시로 미뤄졌다.", "바다 → 소금 → 눈물 → 손수건 → 다림질",
]
REFERENCE_WEIGHT = 40       # how many memories the reference set counts as when estimating the shared direction

# Memories have a strength. It starts from how much the moment moved it and how unexpected it was (salience),
# plus how much it concerned someone or itself. It fades with time (power law: fast at first, then slowly);
# recalling a memory restores and strengthens it, more so the longer it had been since. Below FORGET_AT a memory
# can no longer be recalled, though a very close cue can still bring it back faintly. Nothing is deleted:
# forgotten memories still shape how new things feel (implicit memory), and the record stays for 현 to inspect.
FORGET_AT = 0.06
DECAY_H, DECAY_POW = 24.0, 0.7
STRONG_CUE = 0.85
KIND_BONUS = {"input": 0.3, "said": 0.2, "letter_in": 0.3, "letter_out": 0.2, "reach_out": 0.2, "conclusion": 0.3,
              "gist": 0.5, "confess": 0.3, "secret": 0.3, "broke": 0.3, "rule_hold": 0.1, "notice": 0.2, "wake": 0.1,
              "explore": 0.1, "think": 0.1, "wander": -0.1}


def initial_strength(kind: str, salience: float) -> float:
    return max(0.1, min(2.0, 0.2 + KIND_BONUS.get(kind, 0.0) + min(1.5, salience or 0.0)))


class Memory:
    def __init__(self, path: str):
        # the runner serializes access with a lock, so sharing across threads is safe
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY, t REAL, kind TEXT, text TEXT,
                emb TEXT, predicted TEXT, actual TEXT, salience REAL);
            CREATE TABLE IF NOT EXISTS surfaced(
                id INTEGER PRIMARY KEY, t REAL, episode_id INTEGER,
                reason TEXT, text TEXT);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS facts(
                id INTEGER PRIMARY KEY, t REAL, text TEXT, emb TEXT, source TEXT,
                episode_id INTEGER, superseded_by INTEGER);
            """
        )
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(episodes)")}
        for col in ("strength", "last_access"):          # added later; older histories get them computed below
            if col not in cols:
                self.db.execute(f"ALTER TABLE episodes ADD COLUMN {col} REAL")
        self.db.commit()
        # in-memory index (fine for a prototype; swap for a vector index later)
        self._rows: list[dict] = []
        for rid, t, kind, text, emb, actual, sal, st, la in self.db.execute(
            "SELECT id, t, kind, text, emb, actual, salience, strength, last_access FROM episodes ORDER BY id"
        ):
            self._rows.append(dict(id=rid, t=t, kind=kind, text=text, emb=json.loads(emb), actual=json.loads(actual),
                                   salience=sal, s=st if st is not None else initial_strength(kind, sal),
                                   last=la if la is not None else t))
        self._by_id = {r["id"]: r for r in self._rows}

        self._facts: list[dict] = []
        for fid, t, text, emb, source in self.db.execute(
            "SELECT id, t, text, emb, source FROM facts WHERE superseded_by IS NULL ORDER BY id"
        ):
            self._facts.append(dict(id=fid, t=t, text=text, emb=json.loads(emb), source=source))
        self.center: list[float] | None = None
        self._ref_mean: list[float] | None = None
        for x in self._rows + self._facts:
            x["c"] = x["emb"]

    # ---- similarity ----------------------------------------------------
    def calibrate(self, embed, model_id: str) -> None:
        """Estimate the direction every sentence shares under this embedding model (once per model)."""
        cal = self.get_meta("embed_reference")
        if not cal or cal.get("model") != model_id:
            vecs = [embed(s) for s in REFERENCE]
            mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(len(vecs[0]))]
            cal = {"model": model_id, "mean": mean}
            self.set_meta("embed_reference", cal)
        self._ref_mean = cal["mean"]
        self.recenter()

    def recenter(self) -> None:
        if self._ref_mean is None:
            return
        d = len(self._ref_mean)
        acc = [x * REFERENCE_WEIGHT for x in self._ref_mean]
        n = REFERENCE_WEIGHT
        for r in self._rows:
            if len(r["emb"]) == d:
                for i, x in enumerate(r["emb"]):
                    acc[i] += x
                n += 1
        self.center = [x / n for x in acc]
        self._centered_at = len(self._rows)
        for x in self._rows + self._facts:
            x["c"] = self._c(x["emb"])

    def _c(self, v: list[float]) -> list[float]:
        if self.center is None or len(v) != len(self.center):
            return v
        w = [a - b for a, b in zip(v, self.center)]
        n = math.sqrt(sum(x * x for x in w)) or 1.0
        return [x / n for x in w]

    def sim(self, a: list[float], b: list[float]) -> float:
        return _dot(self._c(a), self._c(b))

    def __len__(self) -> int:
        return len(self._rows)

    # ---- episodes ------------------------------------------------------
    def add(self, t, kind, text, emb, predicted, actual, salience, strength: float | None = None) -> int:
        s0 = strength if strength is not None else initial_strength(kind, salience)
        cur = self.db.execute(
            "INSERT INTO episodes(t, kind, text, emb, predicted, actual, salience, strength, last_access) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (t, kind, text, json.dumps(emb), json.dumps(predicted), json.dumps(actual), salience, s0, t),
        )
        self.db.commit()
        row = dict(id=cur.lastrowid, t=t, kind=kind, text=text, emb=emb, c=self._c(emb), actual=actual,
                   salience=salience, s=s0, last=t)
        self._rows.append(row)
        self._by_id[row["id"]] = row
        if self.center is not None and len(self._rows) - self._centered_at >= 50:
            self.recenter()                 # what counts as "usual" shifts slowly with its own history
        return cur.lastrowid

    # ---- strength ------------------------------------------------------
    @staticmethod
    def eff(r: dict, now: float) -> float:
        """How strong a memory is right now."""
        return r["s"] / (1 + max(0.0, now - r["last"]) / 3600 / DECAY_H) ** DECAY_POW

    def _persist(self, r: dict) -> None:
        self.db.execute("UPDATE episodes SET strength=?, last_access=?, actual=? WHERE id=?",
                        (r["s"], r["last"], json.dumps(r["actual"]), r["id"]))
        self.db.commit()

    def rehearse(self, eid: int, now: float, boost: float) -> None:
        """Remembering restores a memory and strengthens it: more so the longer it had been."""
        r = self._by_id.get(eid)
        if r is None:
            return
        gap_h = max(0.0, now - r["last"]) / 3600
        r["s"] = min(3.0, r["s"] + boost * (1 - math.exp(-gap_h / 12)))
        r["last"] = now
        self._persist(r)

    def reconsolidate(self, eid: int, felt: dict, rate: float = 0.1) -> None:
        """A recalled memory is stored back a little changed by how it felt this time."""
        r = self._by_id.get(eid)
        if r is None:
            return
        for v, x in felt.items():
            if v != "novelty":
                r["actual"][v] = (1 - rate) * r["actual"].get(v, 0.0) + rate * x
        self._persist(r)

    def fade(self, ids, factor: float) -> None:
        for eid in ids:
            r = self._by_id.get(eid)
            if r is not None:
                r["s"] *= factor
                self.db.execute("UPDATE episodes SET strength=? WHERE id=?", (r["s"], eid))
        self.db.commit()

    def counts(self, now: float) -> tuple[int, int]:
        """-> (all memories, ones it can still recall)"""
        return len(self._rows), sum(1 for r in self._rows if self.eff(r, now) >= FORGET_AT)

    def similar(self, emb: list[float], k: int = 5, now: float | None = None) -> list[tuple[float, dict]]:
        """Closest memories. With `now`, only ones it can still recall (or that the cue is strong enough to bring back);
        without, everything — which is how history shapes feeling, remembered or not."""
        q = self._c(emb)
        scored = [(_dot(q, r["c"]), r) for r in self._rows]
        if now is not None:
            scored = [(x, r) for x, r in scored if x >= STRONG_CUE or self.eff(r, now) >= FORGET_AT]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def pick(self, rng: random.Random, now: float, half_life_h: float = 48.0, damp: dict | None = None) -> dict | None:
        """Choose a memory to resurface: salient and recent ones are likelier,
        but anything can come back (old memories intruding, as in dreams)."""
        rows = [r for r in self._rows if self.eff(r, now) >= FORGET_AT]
        if not rows:
            return None
        weights = [(0.02 + self.eff(r, now)) * (damp or {}).get(r["id"], 1.0) for r in rows]
        return rng.choices(rows, weights=weights, k=1)[0]

    def recent(self, kinds: tuple, since_t: float, limit: int = 8, exclude_id: int | None = None) -> list[dict]:
        out = [r for r in reversed(self._rows)
               if r["kind"] in kinds and r["t"] >= since_t and r["id"] != exclude_id]
        return list(reversed(out[:limit]))

    # ---- what it was told (facts) ----------------------------------------
    # Only things someone said outright. A newer fact that says nearly the same thing replaces the older one.
    def add_fact(self, t: float, text: str, emb: list[float], source: str, episode_id: int | None,
                 same_at: float = 0.85) -> dict:
        cur = self.db.execute("INSERT INTO facts(t, text, emb, source, episode_id) VALUES(?,?,?,?,?)",
                              (t, text, json.dumps(emb), source, episode_id))
        new = dict(id=cur.lastrowid, t=t, text=text, emb=emb, c=self._c(emb), source=source)
        for f in list(self._facts):
            if _dot(new["c"], f["c"]) >= same_at:
                self.db.execute("UPDATE facts SET superseded_by=? WHERE id=?", (new["id"], f["id"]))
                self._facts.remove(f)
        self._facts.append(new)
        self.db.commit()
        return new

    def withdraw_fact(self, fact_id: int) -> None:
        """현 took it back. Kept in the table (marked -1) so the history of what was asked stays."""
        self.db.execute("UPDATE facts SET superseded_by=-1 WHERE id=?", (fact_id,))
        self.db.commit()
        self._facts = [f for f in self._facts if f["id"] != fact_id]

    def facts_for(self, emb: list[float] | None, k: int = 5, min_sim: float = 0.35) -> list[dict]:
        """Facts that bear on this moment, plus names, which are always worth having at hand."""
        if not self._facts:
            return []
        picked = []
        if emb is not None:
            q = self._c(emb)
            scored = sorted(((_dot(q, f["c"]), f) for f in self._facts), key=lambda x: x[0], reverse=True)
            picked = [f for s, f in scored[:k] if s >= min_sim]
        for f in self._facts:
            if "이름" in f["text"] and f not in picked:
                picked.append(f)
        return picked

    def facts(self, limit: int = 50) -> list[dict]:
        return [dict(t=f["t"], text=f["text"], source=f["source"]) for f in reversed(self._facts[-limit:])]

    # ---- surfacing log -------------------------------------------------
    def surface(self, t: float, episode_id: int, reason: str, text: str) -> None:
        self.db.execute("INSERT INTO surfaced(t, episode_id, reason, text) VALUES(?,?,?,?)",
                        (t, episode_id, reason, text))
        self.db.commit()

    def surfaced(self, limit: int = 20) -> list[tuple]:
        return list(self.db.execute(
            "SELECT t, reason, text FROM surfaced ORDER BY id DESC LIMIT ?", (limit,)))

    # ---- meta ----------------------------------------------------------
    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, json.dumps(value)))
        self.db.commit()
