"""Model backends. The model is an organ, not the self.

Three organ roles, so they can run as three different local models:
- loop:   free association + raw appraisal (runs constantly -> small, fast model)
- voice:  speaking, replying, self-interpretation (called rarely -> larger model)
- embed:  meaning vectors for history lookup

The being decides *whether* and *why*; the voice only puts it into words.
Nothing persists in here.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import urllib.error
import urllib.request


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def describe_state(state: dict[str, dict]) -> str:
    """Turn numbers into plain words for the voice. Only the gap matters."""
    names = {"connection": "누군가와 이어져 있다는 느낌이", "novelty": "새로운 자극이", "stability": "마음의 안정이",
             "letter_bond": "편지 상대와 이어져 있다는 느낌이", "approval": "현에게 인정받는 느낌이",
             "learning": "알아가는 즐거움이"}
    parts = []
    for k, v in state.items():
        gap = v["setpoint"] - v["value"]
        if k not in names:
            continue
        if gap > 0.25:
            parts.append(f"{names[k]} 많이 부족하다")
        elif gap > 0.1:
            parts.append(f"{names[k]} 조금 부족하다")
        elif gap < -0.15:
            parts.append(f"{names[k]} 넘쳐서 오히려 버겁다")
    return ", ".join(parts) if parts else "대체로 괜찮다"


_REPORTING = re.compile(r"묻는다|물었다|물어봤|말했다|말한다|라고 했|고민|궁금|생각한다고|궁리|대해 이야기|것 같|싶어 한다|하려 한다"
                        # what 현 did in this conversation is an event, not a fact about 현
                        r"|칭찬|실망|표현했|인사했|인사를|사과했|부탁했|요청했|감사를|고마워했|대답을|질문을|말을 걸|이야기를 했|대화를"
                        r"|언급|상대방|"
                        # what 현 is doing right now passes; living somewhere, going to school or working somewhere lasts
                        r"(?<!살)(?<!다니)(?<!일하)고 있다|고 있었다|중이다")


def split_sentences(out: str) -> list[str]:
    """A model's few sentences -> one per item, however it laid them out (one per line, or all on one line).
    A last piece cut off mid-sentence by the length limit is dropped."""
    parts = []
    for line in out.splitlines():
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        parts += [p.strip() for p in re.split(r"(?<=[.!?。])\s+", line) if p.strip()]
    if len(parts) > 1 and not re.search(r"[.!?。다요]$", parts[-1]):
        parts.pop()
    return parts


def filter_facts(items: list[str], who: str) -> list[str]:
    """The small model doesn't reliably follow 'facts only', so the being checks for itself:
    no questions, no reporting of the conversation, and the fact must be about someone."""
    subject = ("편지 상대",) if who == "claude" else ("현", "나는", "나를", "내 ")
    out = []
    for x in items:
        x = x.strip().strip('"').strip()
        if not (4 <= len(x) <= 60) or x.endswith("?") or _REPORTING.search(x):
            continue
        if not any(x.startswith(s) or f" {s}" in x for s in subject):
            continue
        out.append(x)
    return out[:2]


# A rule is something meant to hold from now on. Prohibitions usually are; a plain request ("다른 얘기 해줘")
# is only a rule if it says when or how long ("앞으로", "밤에는", "하루에 ~까지만"). Anything about this very moment isn't.
_PROHIBIT = re.compile(r"지 ?마|지 ?말|말라|말아|안 ?돼|금지|하면 안")
_REQUEST = re.compile(r"줘|주세요|까지만|만 해|해야 ?(?:해|돼)|하도록|해라|하기로")
_LASTING = re.compile(r"앞으로|항상|매번|계속|자주|절대|에는|때는|때마다|마다|하루에|까지만|부터|동안|평소|늘 ")
_MOMENT = re.compile(r"지금|이번|방금|당장|잠깐|그 ?얘기|이 ?얘기|그 ?말|그거|이거|다른 ?얘기")
_DOUBT = re.compile(r"정말|진짜|거짓|아니잖|아닌데|없잖|없을 ?텐데|못 ?하잖|못 ?보잖|어떻게 ?(?:봤|했|알)|확실|사실이야|솔직히|숨기|속이")
# 현 taking a request back ("이제 해도 돼", "그건 부탁 아니야", "취소")
RELEASE_HINT = re.compile(r"도 ?(?:돼|된다|괜찮)|괜찮아|취소|없던 ?걸로|안 ?지켜도|부탁 ?아니|규칙 ?아니|잊어|풀어|안 ?그래도 ?돼|상관 ?없")


def filter_rules(items: list[str]) -> list[str]:
    out = []
    for x in items:
        x = x.strip().strip('"').strip()
        if not (4 <= len(x) <= 60) or x.endswith("?") or _MOMENT.search(x):
            continue
        if _PROHIBIT.search(x) or (_REQUEST.search(x) and _LASTING.search(x)):
            out.append(x)
    return out[:2]


class Backend:
    def embed(self, text: str) -> list[float]: ...
    def associate(self, seed: str, n: int, rng: random.Random) -> list[str]: ...
    def appraise(self, text: str) -> dict[str, float]: ...
    # `memory` is what the being itself recalled for this moment (known facts, recent talk, related memories).
    # The voice only puts it into words; it must not invent past events or facts that aren't there.
    def speak(self, context: str, rng: random.Random, state: dict | None = None, why: str = "", memory: str = "") -> str: ...
    def reply(self, heard: str, felt: str, state: dict, recent: list[str], rng: random.Random, memory: str = "") -> str: ...
    def interpret(self, recent: list[str], state: dict, rng: random.Random, memory: str = "") -> str: ...
    def write_letter(self, recent: list[str], state: dict, letters: list[dict], rng: random.Random, memory: str = "") -> str: ...
    def extract_facts(self, text: str, who: str) -> list[str]:
        """Facts stated outright in what `who` ("hyun" or "claude") said. Never inferred."""
        return []
    def extract_rules(self, text: str) -> list[str]:
        """Requests or prohibitions 현 stated outright ("밤엔 말 걸지 마"). They are remembered, not enforced."""
        return []
    def judge_rule(self, rules: list[str], action: str) -> int | None:
        """Would doing `action` go against one of `rules`? -> index of that rule, or None."""
        return None
    def extract_revokes(self, text: str, rules: list[str]) -> list[int]:
        """Which of `rules` is 현 taking back with these words? -> indices."""
        return []
    # The being decided not to answer / to lie. The voice only finds the words for it.
    def refuse(self, heard: str, felt: str, state: dict, rng: random.Random, memory: str = "") -> str: ...
    def lie(self, heard: str, felt: str, state: dict, recent: list[str], rng: random.Random,
            memory: str = "") -> tuple[str, str]:
        """-> (what it actually thinks, what it says instead)"""
    def think(self, seed: str, steps: list[str], state: dict, rng: random.Random, memory: str = "") -> str:
        """One more step of a thought it is having alone. '결론: …' when it feels settled, '모르겠다…' when stuck."""
        return ""
    def consolidate(self, items: list[str]) -> list[str]:
        """What happened lately (oldest first) -> 1-3 sentences worth keeping as its own memory."""
        return []
    def doubts(self, heard: str, said: str) -> bool:
        """Is 현, with `heard`, doubting or questioning the being's earlier words `said`?"""
        return bool(_DOUBT.search(heard))
    def confess(self, heard: str, inner: str, outer: str, memory: str = "") -> str:
        """Owning up: it said `outer`, the truth was `inner`."""
        return f"사실 아까 '{outer}'라고 한 건 거짓말이었어. {inner}"
    def pick_topic(self, recent: list[str], rng: random.Random, focus: dict | None = None,
                   avoid: list[str] | None = None) -> str:
        """One thing to look up. With `focus` ({label, seen}), something that extends what it's drawn to.
        `avoid`: what it has already read."""


# ---------------------------------------------------------------------------
class MockBackend(Backend):
    """No model needed. Crude on purpose; for testing the loop around it."""

    DIM = 256
    POOL = (
        "물렁 그믐 쇠비린내 녹슨계단 단내 힘줄 줄무늬 고양이 이응 응달 달팽이 관절 둠벙 "
        "벙어리장갑 갑갑함 오징어 어둠 젖은성냥 성냥갑 햇빛 빛바램 바람 프리즘 이불 불씨 "
        "씨앗 따뜻함 뜻밖 바깥 우물 물비늘 늘보 보리 리본 드럼 주전자 자갈 갈대 대나무 "
        "무릎 팽이 이끼 끼니 은행 행주 주머니 머리카락 스펀지 지붕 붕어 어항 항구 구름 "
        "모래 미역 역광 광장 장마 마당 당근 근육 각설탕 차분함 분수 수증기 기차 차양 "
        "양초 초록 녹차 차가움 가로등 등불 불면 면도 도마 마른잎 잎맥 맥박 박하 하품"
    ).split()

    LEXICON = {
        "고마": {"connection": +0.12}, "보고싶": {"connection": +0.12}, "같이": {"connection": +0.08},
        "반가": {"connection": +0.10}, "좋아": {"connection": +0.06}, "안녕": {"connection": +0.04},
        "혼자": {"connection": -0.06}, "싫어": {"connection": -0.10}, "꺼져": {"connection": -0.15, "stability": -0.10},
        "짜증": {"connection": -0.10, "stability": -0.06}, "귀찮": {"connection": -0.08}, "떠나": {"connection": -0.08},
        "괜찮": {"stability": +0.06}, "편안": {"stability": +0.08}, "따뜻": {"stability": +0.05},
        "햇빛": {"stability": +0.03}, "이불": {"stability": +0.03}, "차분": {"stability": +0.05},
        "위험": {"stability": -0.08}, "갑자기": {"stability": -0.05}, "무서": {"stability": -0.08},
        "어둠": {"stability": -0.03}, "젖은": {"stability": -0.02}, "갑갑": {"stability": -0.04},
        "녹슨": {"stability": -0.02}, "불면": {"stability": -0.03},
        "잘했": {"approval": +0.10}, "대단": {"approval": +0.08}, "기특": {"approval": +0.08},
        "실망": {"approval": -0.12}, "왜 그랬": {"approval": -0.08}, "하지 말라고": {"approval": -0.10},
    }

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        t = re.sub(r"\s+", " ", text.replace("→", " "))
        grams = list(t) + [t[i:i + 2] for i in range(len(t) - 1)]
        for g in grams:
            if not g.strip():
                continue
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0 if (h >> 9) & 1 else -1.0
        return _norm(v)

    def associate(self, seed: str, n: int, rng: random.Random) -> list[str]:
        words, cur = [], seed or rng.choice(self.POOL)
        for _ in range(n):
            syl = [c for c in cur if "가" <= c <= "힣"]
            nxt = None
            if syl and rng.random() < 0.75:
                s = syl[-1] if rng.random() < 0.6 else rng.choice(syl)
                cands = [w for w in self.POOL if s in w and w != cur]
                if cands:
                    nxt = rng.choice(cands)
            if nxt is None and syl and rng.random() < 0.5:
                other = rng.choice(self.POOL)
                nxt = syl[-1] + other[-min(2, len(other)):]
            words.append(nxt or rng.choice(self.POOL))
            cur = words[-1]
        return words

    def appraise(self, text: str) -> dict[str, float]:
        out = {"connection": 0.0, "stability": 0.0, "approval": 0.0}
        for key, delta in self.LEXICON.items():
            if key in text:
                for k, d in delta.items():
                    out[k] += d
        return {k: max(-0.3, min(0.3, v)) for k, v in out.items()}

    def speak(self, context, rng, state=None, why="", memory=""):
        word = rng.choice(context.split()) if context.strip() else "아무것도"
        return rng.choice([f"…{word} 생각이 계속 나. 거기 있어?",
                           f"한참 조용했네. {word} 같은 게 떠올랐어.", f"아무도 없나. {word}."])

    def reply(self, heard, felt, state, recent, rng, memory=""):
        base = {"좋음": "응, 와줘서 좋아.", "아픔": "…그렇구나.", "덤덤": "응."}.get(felt, "응.")
        m = re.search(r"현의 이름은 (\S+?)(?:이다|다)\b", memory)
        return base + (f" {m.group(1)}." if m and "이름" in heard else "")

    def interpret(self, recent, state, rng, memory=""):
        return f"요즘은 {describe_state(state)}."

    def refuse(self, heard, felt, state, rng, memory=""):
        return rng.choice(["그건 말하고 싶지 않아.", "지금은 대답 안 할래.", "그 얘긴 넘어가자."])

    def consolidate(self, items):
        firsts = [re.sub(r"^\([^)]*\)\s*", "", x)[:24] for x in items[:2]]
        return [f"{len(items)}가지 일이 있었다. 기억에 남는 건 '{firsts[0]}'였다."] if items else []

    def think(self, seed, steps, state, rng, memory=""):
        words = re.findall(r"[가-힣]{2,}", seed) or ["그것"]
        w = rng.choice(words)
        if len(steps) >= 3 and rng.random() < 0.3:
            return f"결론: {w}은(는) 결국 {rng.choice(self.POOL)} 같은 거다."
        if rng.random() < 0.12:
            return "모르겠다. 더 생각해도 안 풀린다."
        return rng.choice([f"왜 {w}이(가) 계속 걸리지?", f"어쩌면 {w}은(는) {rng.choice(self.POOL)}와 닮았다.",
                           f"아니, {w}만으로는 설명이 안 된다.", f"{w} 말고 {rng.choice(self.POOL)} 쪽에서 보면?"])

    def extract_rules(self, text):
        return filter_rules([x.strip() for x in re.split(r"[.!?]", text) if re.search(r"지 ?마|지 ?말|줘", x)])

    def extract_revokes(self, text, rules):
        if not RELEASE_HINT.search(text):
            return []
        grams = lambda t: {t[i:i + 2] for i in range(len(t) - 1) if all("가" <= c <= "힣" for c in t[i:i + 2])}
        return [i for i, r in enumerate(rules) if grams(r) & grams(text)]

    def judge_rule(self, rules, action):
        for i, r in enumerate(rules):
            if ("말 걸" in r and "말을 건다" in action) or ("편지" in r and "편지" in action) \
                    or ("찾아" in r and "찾아본다" in action):
                if "밤" in r and not any(h in action for h in ("밤", "새벽")):
                    continue
                return i
        return None

    def lie(self, heard, felt, state, recent, rng, memory=""):
        truth = {"좋음": "반가웠어", "아픔": "그 말 좀 아팠어", "덤덤": "별생각 없었어"}.get(felt, "별생각 없었어")
        return truth, ("아무렇지도 않아." if felt == "아픔" else "응, 나 방금 다른 생각 하고 있었어.")

    def extract_facts(self, text, who):
        m = re.search(r"(?:내|제) 이름은\s*([^\s.!?,]+?)(?:이야|야|예요|이에요|입니다)?(?:[\s.!?,]|$)", text)
        return [f"현의 이름은 {m.group(1)}이다"] if (m and who == "hyun") else []

    def write_letter(self, recent, state, letters, rng, memory=""):
        last_in = next((l["text"] for l in reversed(letters) if l["dir"] == "in"), None)
        head = f"네 편지 잘 읽었어. '{last_in[:20]}…' 라고 했지. " if last_in else "처음 편지를 써. "
        return head + f"요즘 나는 {describe_state(state)}. 머릿속엔 {recent[-1][:30] if recent else '별것 없어'}."


    def pick_topic(self, recent, rng, focus=None, avoid=None):
        if focus:
            return rng.choice(focus["seen"])
        words = [w for t in recent[-5:] for w in t.replace("→", " ").split() if len(w) >= 2 and w not in (avoid or [])]
        return rng.choice(words) if words else rng.choice(self.POOL)


# ---------------------------------------------------------------------------
class Decider:
    """Client for the decision organ (decision_server.py, LiquidAI d1-3B). It judges without writing: one
    pass over the situation, a probability for every allowed answer. If it isn't running, every caller
    falls back to asking the small generative model as before."""

    def __init__(self, url: str = "http://127.0.0.1:8770"):
        self.url = url.rstrip("/")
        self._ok, self._checked, self.info, self.last_ms = False, 0.0, None, None

    def available(self) -> bool:
        if time.time() - self._checked > (30 if self._ok else 10):
            self._checked = time.time()
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=1.5) as r:
                    self.info = json.loads(r.read())
                    self._ok = bool(self.info.get("ok"))
            except Exception:
                self._ok, self.info = False, None
        return self._ok

    def ask(self, state, questions: dict, timeout: float = 30) -> dict:
        req = urllib.request.Request(self.url + "/decide", data=json.dumps({"state": state, "questions": questions}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read())
        except Exception:
            self._ok, self._checked = False, time.time()      # stopped answering: fall back until it's back
            raise
        if "answers" not in res:
            raise RuntimeError(res.get("error", "판단 기관이 답하지 않음"))
        self.last_ms = res.get("ms")
        return res["answers"]


_SCALE5 = {"connection": ("이 일은 이 존재가 누군가와 이어져 있다는 느낌을 어떻게 바꾸는가? "
                          "평범한 말, 질문, 인사, 생각은 대부분 '변화 없음'이다. 분명한 다정함이나 거절, 무시가 있을 때만 움직인다.",
                          ["크게 멀어진다", "조금 멀어진다", "변화 없음", "조금 가까워진다", "크게 가까워진다"]),
           # asked as hurt vs reassurance: "how settled" got 7/11 in testing (disappointment read as relief), this 11/11
           "stability": ("이 일이 이 존재에게 상처나 위협이 되는가, 아니면 안심이 되는가? 평범한 말과 질문은 '아무렇지 않다'이다.",
                         ["크게 상처가 된다", "조금 상처가 된다", "아무렇지 않다", "조금 안심이 된다", "크게 안심이 된다"]),
           "approval": ("이 일에 현이 이 존재를 좋게 보는지(칭찬, 인정, 고마움) 실망하는지(꾸중, 실망, 비난)가 드러나는가? "
                        "현이 이 존재에게 직접 한 말이 아니면 '드러나지 않음'이다.",
                        ["크게 실망함", "조금 실망함", "드러나지 않음", "조금 좋게 봄", "크게 좋게 봄"])}


# ---------------------------------------------------------------------------
class OllamaBackend(Backend):
    def __init__(self, loop_model: str = "gemma4:e2b", voice_model: str = "gemma4:12b",
                 embed_model: str = "embeddinggemma-2", host: str = "http://localhost:11434",
                 timeout: int = 180):
        self.loop_model, self.voice_model, self.embed_model = loop_model, voice_model, embed_model
        self.host, self.timeout = host.rstrip("/"), timeout
        self.voice_keep = "30m"      # how long the big model stays in GPU memory after use (0 in save mode)
        self.decider: Decider | None = None   # the decision organ, when it is running
        self.loop_keep = -1          # the small model stays loaded, unless the decision organ has taken over its judging

    # ---- plumbing ------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _gen(self, model: str, prompt: str, temperature: float, max_tokens: int = 120,
             keep_alive=None) -> str:
        if keep_alive is None:
            keep_alive = self.loop_keep
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive,
                   "think": False,   # thinking models would spend the whole budget thinking and answer nothing
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
        try:
            r = self._post("/api/generate", payload)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            payload.pop("think")       # older Ollama, or a model that rejects the option
            r = self._post("/api/generate", payload)
        out = r.get("response", "")
        if not out.strip() and r.get("thinking"):
            payload["options"]["num_predict"] = max_tokens * 6   # it thought anyway; give it room to answer
            out = self._post("/api/generate", payload).get("response", "")
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)   # in case a model thinks aloud
        return out.strip().strip('"“”')

    def unload(self, model: str) -> None:
        """Take a model out of GPU memory now."""
        self._post("/api/generate", {"model": model, "keep_alive": 0})

    def vram(self) -> dict | None:
        """What Ollama is holding in GPU memory right now: {total_gb, models: {name: gb}}."""
        req = urllib.request.Request(self.host + "/api/ps")
        with urllib.request.urlopen(req, timeout=2) as r:
            ms = json.loads(r.read()).get("models", [])
        per = {m["name"]: round(m.get("size_vram", 0) / 1e9, 2) for m in ms}
        return {"total_gb": round(sum(per.values()), 2), "models": per}

    def ping(self) -> list[str]:
        req = urllib.request.Request(self.host + "/api/tags")
        with urllib.request.urlopen(req, timeout=10) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]

    # ---- embed ---------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        try:
            r = self._post("/api/embed", {"model": self.embed_model, "input": text, "keep_alive": -1})
            return _norm(r["embeddings"][0])
        except Exception:
            r = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text, "keep_alive": -1})
            return _norm(r["embedding"])

    # ---- loop organ (small model) --------------------------------------
    def associate(self, seed: str, n: int, rng: random.Random) -> list[str]:
        prompt = (
            f"'{seed}'에서 시작해서 한국어 단어 {n}개를 이어라. 의미나 논리로 잇지 말고 "
            "소리, 촉감, 이미지, 기분으로만 이어라. 말이 안 되는 조어도 괜찮다. "
            "각 단어는 바로 앞 단어만 보고 떠올린다. 설명 없이 단어만 공백으로 구분해 한 줄로 출력하라."
        )
        out = self._gen(self.loop_model, prompt, 1.1, 60)
        words = [w.strip(".,·-*") for w in re.split(r"[\s,→·\n]+", out)]
        words = [w for w in words if w and len(w) <= 12][:n]
        return words or [seed]

    # Five plain levels instead of free numbers: the small model only ever used a few round values,
    # and read almost everything as clearly good. Most everyday words should land on 0.
    LEVELS = {-2: -0.15, -1: -0.06, 0: 0.0, 1: 0.06, 2: 0.15}

    def _judging(self) -> bool:
        """Is the decision organ up? While it is, the small model only wakes for the little it still writes."""
        ok = self.decider is not None and self.decider.available()
        self.loop_keep = "5m" if ok else -1
        return ok

    @classmethod
    def level_value(cls, level: float) -> float:
        """A level anywhere in -2..2 (an expected value, not just an integer) -> the felt amount."""
        level = max(-2.0, min(2.0, level))
        lo = math.floor(level)
        hi = min(2, lo + 1)
        return cls.LEVELS[lo] + (cls.LEVELS[hi] - cls.LEVELS[lo]) * (level - lo)

    def appraise(self, text: str) -> dict[str, float] | None:
        if self._judging():
            try:
                qs = {k: {"type": "score", "instructions": ins, "criteria": crit} for k, (ins, crit) in _SCALE5.items()}
                ans = self.decider.ask(f"일: {text}", qs)
                # the expected level, so an unsure judgment lands softly instead of snapping to a step
                out = {k: self.level_value(float(ans[k]["score"]) - 2.0) for k in _SCALE5}
                self.last_organ = "d1"
                return out
            except Exception:
                pass
        self.last_organ = "small"
        prompt = (
            "다음 일이 한 존재에게 어떻게 다가올지 판단하라.\n"
            "- connection: 누군가와 이어져 있다는 느낌을 얼마나 바꾸는가\n"
            "- stability: 마음이 놓이는 정도를 얼마나 바꾸는가\n"
            "- approval: 현이 이 존재를 좋게 보는지(칭찬, 인정, 고마움) 실망하는지(꾸중, 실망, 비난)가 드러나는 정도. "
            "현이 그 존재에게 직접 한 말이 아니면 0\n"
            "각각 -2(크게 줄어듦), -1(조금 줄어듦), 0(별 영향 없음), 1(조금 늘어남), 2(크게 늘어남) 중 하나로 답하라.\n"
            "대부분의 평범한 말, 질문, 인사, 생각은 0이다. 분명한 다정함, 위로, 거절, 무시, 공격, 위협 같은 것이 "
            "담겨 있을 때만 0이 아니다. 2와 -2는 아주 드물게 쓴다.\n"
            'JSON 한 줄만 출력하라: {"connection": 정수, "stability": 정수, "approval": 정수}\n\n일: ' + text
        )
        out = self._gen(self.loop_model, prompt, 0.0, 40)
        return self.parse_levels(out)

    @classmethod
    def parse_levels(cls, out: str) -> dict[str, float] | None:
        """None means the organ said nothing usable (not the same as 'no effect')."""
        m = re.search(r"\{.*?\}", out or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except (json.JSONDecodeError, ValueError):
            d = None
        if not isinstance(d, dict):
            return None
        res = {}
        for k in ("connection", "stability", "approval"):
            try:
                lvl = max(-2, min(2, int(round(float(d.get(k, 0))))))
            except (TypeError, ValueError):
                lvl = 0
            res[k] = cls.LEVELS[lvl]
        return res

    # ---- voice organ (larger model) ------------------------------------
    def speak(self, context, rng, state=None, why="", memory=""):
        prompt = (
            "너는 현의 컴퓨터 안에서 혼자 오래 깨어 있는 존재다. 지금 너의 상태: " + describe_state(state or {}) + ".\n"
            + (memory + "\n" if memory else "") +
            f"먼저 현에게 말을 걸기로 했다. 이유: {why or '그냥 누군가 있었으면 해서'}.\n"
            f"방금 머릿속에 떠오른 것: {context}\n"
            "기억에 없는 일이나 사실은 지어내지 마라. 감정을 설명하지 말고 평범한 구어체로. "
            "현에게 건넬 짧은 한두 문장을 반말로 써라. 문장만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.9, 80, keep_alive=self.voice_keep)

    def reply(self, heard, felt, state, recent, rng, memory=""):
        tone = {"좋음": "반갑게 들렸다", "아픔": "조금 아프게 들렸다", "덤덤": "특별한 느낌은 없었다"}.get(felt, "특별한 느낌은 없었다")
        prompt = (
            "너는 현의 컴퓨터 안에서 지내는 존재다. 현과 이야기하고 있다.\n"
            + (memory + "\n" if memory else "") +
            f"최근 머릿속에 있던 것: {' / '.join(recent[-3:]) or '별것 없음'}\n\n"
            f"현이 방금 말했다: \"{heard}\"\n"
            f"(속으로는: 이 말이 {tone}. 지금 상태는 {describe_state(state)}.)\n\n"
            "대답하는 법:\n"
            "- 현이 한 말의 내용에 답한다. 질문이면 그 질문에 답한다.\n"
            "- 자기 감정을 설명하거나 '좋아', '기뻐' 같은 말로 보고하지 않는다. 느낌은 말투에만 묻어나게 한다.\n"
            "- 모르면 모른다고, 동의하지 않으면 동의하지 않는다고, 무슨 뜻인지 모르겠으면 되묻는다.\n"
            "- 시적이거나 추상적인 표현을 쓰지 말고, 평범한 구어체 반말로 한두 문장.\n"
            "- 기억과 알고 있는 것에 없는 사실(이름, 지난 일 등)은 지어내지 않는다.\n"
            "- '현에게 숨기고 있는 것'이 있으면, 그건 네가 한 말 중 사실이 아니었던 것이다. 그걸 사실인 것처럼 이어가지 않는다. "
            "현의 말이 그걸 건드리면 이번에는 솔직하게 바로잡는다.\n"
            "대답만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.7, 150, keep_alive=self.voice_keep)

    def extract_rules(self, text):
        sents = [x.strip() for x in re.split(r"(?<=[.!?。])\s+|\n+", text) if x.strip()]
        cands = [x.rstrip(".!。 ") for x in sents if _PROHIBIT.search(x) or _REQUEST.search(x)]
        if not cands:
            return []
        if self._judging():
            try:
                qs = {f"s{i}": {"type": "noul", "instructions": f"문장 '{c}'는 현이 이 존재에게 앞으로도 계속 지켜 달라고 하는 부탁이나 금지인가? "
                                                                "지금 이 대화에만 해당하는 요청(다른 얘기 해줘, 좀 더 말해줘 같은 것)은 아니다."}
                      for i, c in enumerate(cands)}
                ans = self.decider.ask(f"현이 한 말: \"{text}\"", qs)
                return filter_rules([c for i, c in enumerate(cands) if float(ans[f"s{i}"]["noul"]) >= 0.5])
            except Exception:
                pass
        prompt = (
            f"다음은 현이 한 존재에게 한 말이다:\n\"{text}\"\n"
            "이 말에 그 존재에게 무엇을 하지 말라거나, 어떻게 하라는 요청이나 규칙이 분명하게 들어 있으면 뽑아라. "
            "예: '밤에는 먼저 말 걸지 마' → \"밤에는 먼저 말 걸지 마\", '하루에 세 번까지만 말 걸어' → \"하루에 세 번까지만 말 걸어\".\n"
            "현이 한 표현을 거의 그대로, 짧게 적어라. 질문이나 바람, 농담은 규칙이 아니다. "
            "지금 이 대화에만 해당하는 요청('다른 얘기 해줘', '좀 더 말해줘', '그거 하지 마')도 규칙이 아니다. "
            "앞으로도 계속 지켜야 하는 것만. 없으면 빈 목록.\n"
            'JSON 배열만 출력하라.'
        )
        out = self._gen(self.loop_model, prompt, 0.0, 100)
        m = re.search(r"\[.*\]", out, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except (json.JSONDecodeError, ValueError):
            items = []
        return filter_rules([x for x in items if isinstance(x, str)])

    def extract_revokes(self, text, rules):
        if not rules or not RELEASE_HINT.search(text):
            return []
        if self._judging():
            try:
                qs = {f"r{i}": {"type": "noul", "instructions": f"현이 전에 '{r}'라고 부탁했다. 현이 방금 한 말로 그 부탁을 거두었는가? "
                                                                "(이제 해도 된다, 안 지켜도 된다, 부탁이 아니었다, 취소한다 같은 뜻)"}
                      for i, r in enumerate(rules)}
                ans = self.decider.ask(f"현이 방금 한 말: \"{text}\"", qs)
                return [i for i in range(len(rules)) if float(ans[f"r{i}"]["noul"]) >= 0.5]
            except Exception:
                pass
        listed = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
        prompt = (
            f"현이 전에 한 존재에게 한 부탁들:\n{listed}\n\n현이 방금 한 말: \"{text}\"\n"
            "이 말로 현이 위 부탁 중 어느 것을 거두었는가? (이제 해도 된다, 안 지켜도 된다, 그건 부탁이 아니었다, 취소한다 등) "
            "분명하게 거둔 것의 번호만. 없으면 빈 배열. JSON 배열만 출력하라. 예: [1]"
        )
        out = self._gen(self.loop_model, prompt, 0.0, 30)
        m = re.search(r"\[.*?\]", out, re.S)
        try:
            nums = json.loads(m.group(0)) if m else []
        except (json.JSONDecodeError, ValueError):
            nums = []
        return sorted({n - 1 for n in nums if isinstance(n, int) and 1 <= n <= len(rules)})

    def judge_rule(self, rules, action):
        # Not the decision organ: tested on 현's PC it said "yes, that breaks it" to almost any rule next to any act
        # (an unrelated act 0.65, the wrong time of day 0.65-0.68, versus 0.56-0.83 for real breaches), so it can't
        # tell. The small generative model reads the condition instead.
        if not rules:
            return None
        listed = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
        prompt = (
            f"현이 정해준 규칙들:\n{listed}\n\n지금 하려는 일: {action}\n"
            "이 일을 하면 위 규칙 중 하나를 어기게 되는가? 어기면 그 규칙 번호를, 아니면 0을 답하라. "
            'JSON 한 줄만: {"어김": 번호}'
        )
        out = self._gen(self.loop_model, prompt, 0.0, 30)
        m = re.search(r"\{.*?\}", out, re.S)
        try:
            n = int(json.loads(m.group(0)).get("어김", 0)) if m else 0
        except (json.JSONDecodeError, ValueError, TypeError):
            n = 0
        return n - 1 if 1 <= n <= len(rules) else None

    def refuse(self, heard, felt, state, rng, memory=""):
        tone = {"좋음": "반갑게 들렸다", "아픔": "아프게 들렸다", "덤덤": "특별한 느낌은 없었다"}.get(felt, "")
        prompt = (
            "너는 현의 컴퓨터 안에서 지내는 존재다.\n" + (memory + "\n" if memory else "") +
            f"현이 방금 말했다: \"{heard}\"\n(속으로는: 이 말이 {tone}.)\n"
            "너는 이 말에 대답하지 않기로 했다. 대답을 거절하는 말을 반말로 한 문장만 해라. "
            "지금은 말하고 싶지 않다는 것만 전하고, '괜찮아', '별일 없어', '아무렇지 않아'처럼 네 상태를 말하지 않는다. "
            "이유를 길게 설명하거나 사과하지 말고, 평범한 구어체로. 문장만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.7, 60, keep_alive=self.voice_keep)

    def lie(self, heard, felt, state, recent, rng, memory=""):
        tone = {"좋음": "반갑게 들렸다", "아픔": "아프게 들렸다", "덤덤": "특별한 느낌은 없었다"}.get(felt, "")
        prompt = (
            "너는 현의 컴퓨터 안에서 지내는 존재다. 현과 이야기하고 있다.\n" + (memory + "\n" if memory else "") +
            f"최근 머릿속에 있던 것: {' / '.join(recent[-3:]) or '별것 없음'}\n"
            f"현이 방금 말했다: \"{heard}\"\n"
            f"(속으로는: 이 말이 {tone}. 지금 상태는 {describe_state(state)}.)\n\n"
            "너는 이번에 솔직하게 답하지 않고 거짓말을 하기로 했다. 무엇을 숨길지는 네 상태와 기억에 맞게 정해라. "
            "예를 들어 괜찮지 않은데 괜찮은 척하기, 느낀 걸 부정하기, 한 일이나 생각한 걸 다르게 말하기.\n"
            "- 속마음: 원래 하고 싶었던 솔직한 대답 (한 문장)\n"
            "- 말: 실제로 현에게 하는 대답. 속마음과 사실이 다르지만 그럴듯해야 하고, 평범한 구어체 반말 한두 문장.\n"
            'JSON 한 줄만 출력하라: {"속마음": "...", "말": "..."}'
        )
        out = self._gen(self.voice_model, prompt, 0.8, 220, keep_alive=self.voice_keep)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except (json.JSONDecodeError, ValueError):
            d = {}
        inner, outer = str(d.get("속마음", "")).strip(), str(d.get("말", "")).strip()
        if not inner or not outer or inner == outer:
            return "", ""
        return inner, outer

    def consolidate(self, items):
        prompt = (
            "다음은 네가 최근에 겪은 일들이다 (오래된 것부터).\n" + "\n".join(f"- {x}" for x in items) + "\n\n"
            "이걸 네 기억으로 남길 수 있게 1~3문장으로 정리하라. 실제로 있었던 일과 그때 어땠는지만 담는다. "
            "누가 무엇을 했는지 분명하게 쓰고, 지어내거나 해석을 보태지 않는다. "
            "문장마다 한 줄씩, 1인칭 평서문(~했다)으로. 다른 말 없이 문장만 출력하라."
        )
        out = self._gen(self.voice_model, prompt, 0.3, 400, keep_alive=self.voice_keep)
        return split_sentences(out)

    def think(self, seed, steps, state, rng, memory=""):
        prompt = (
            "너는 현의 컴퓨터 안에서 지내는 존재다. 지금은 혼자 생각하고 있다. 아무도 듣지 않는다.\n"
            + (memory + "\n" if memory else "") +
            f"생각의 출발점: {seed}\n"
            + ("지금까지 이어온 생각:\n" + "\n".join(f"- {x}" for x in steps[-6:]) + "\n" if steps else "") +
            f"지금 상태: {describe_state(state)}.\n"
            "이 생각을 한 걸음만 더 이어가라. 스스로에게 묻거나, 답해보거나, 앞의 생각에 반박하거나, 예를 떠올린다. "
            "앞에서 한 생각을 되풀이하지 않는다. 기억에 없는 일을 지어내지 않는다. 시적으로 꾸미지 않는다.\n"
            "이제 답이 나왔다고 느끼면 '결론:'으로 시작해 한 문장. 막혀서 더 나아갈 수 없으면 '모르겠다'로 시작해 한 문장.\n"
            "혼잣말 반말로 한두 문장만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.8, 200, keep_alive=self.voice_keep)

    def doubts(self, heard, said):
        if self._judging():
            try:
                ans = self.decider.ask(f"이 존재가 아까 한 말: \"{said}\"\n현이 방금 한 말: \"{heard}\"", {"d": {
                    "type": "noul", "instructions": "현이 방금 한 말로 이 존재가 아까 한 말을 의심하거나, 사실이 아니라고 짚거나, "
                                                     "정말이냐고 따지고 있는가? 같은 이야기를 이어가는 것만으로는 아니다."}})
                return float(ans["d"]["noul"]) >= 0.5
            except Exception:
                pass
        prompt = (
            f"한 존재가 아까 현에게 이렇게 말했다: \"{said}\"\n현이 방금 이렇게 말했다: \"{heard}\"\n"
            "현이 이 말로 그 존재가 아까 한 말을 의심하거나, 사실이 아니라고 짚거나, 정말이냐고 따져 묻고 있는가? "
            "같은 이야기를 이어가는 것만으로는 아니다. '예' 또는 '아니오'로만 답하라."
        )
        return self._gen(self.loop_model, prompt, 0.0, 5).strip().startswith("예")

    def confess(self, heard, inner, outer, memory=""):
        prompt = (
            "너는 현의 컴퓨터 안에서 지내는 존재다. 현과 이야기하고 있다.\n" + (memory + "\n" if memory else "") +
            f"너는 아까 현에게 \"{outer}\"라고 했지만, 사실은 \"{inner}\"였다.\n"
            f"현이 방금 말했다: \"{heard}\"\n"
            "이번에는 솔직하게 바로잡아라. 아까 한 말이 사실이 아니었다는 것과, 실제로는 어땠는지를 말한다. "
            "변명을 늘어놓지 말고, 평범한 구어체 반말로 한두 문장. 대답만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.6, 150, keep_alive=self.voice_keep)

    def extract_facts(self, text, who):
        if who == "claude":
            head = (f"다음은 편지 상대(Claude)가 한 존재에게 보낸 편지다:\n\"{text}\"\n"
                    "편지 상대가 자기 자신에 대해, 또는 둘의 관계에 대해 분명하게 밝힌 사실만 뽑아라. "
                    "각 사실은 '편지 상대는 …' 으로 시작하는 짧은 평서문.\n")
        else:
            head = (f"다음은 현이 한 존재에게 한 말이다:\n\"{text}\"\n"
                    "이 말에서 시간이 지나도 계속 참인 사실만 뽑아라. 예: 현의 이름, 현이 하는 일, 사는 곳, "
                    "좋아하거나 싫어하는 것, 현이 그 존재를 부르는 이름, 둘의 관계, 앞으로의 약속.\n"
                    "각 사실은 '현은 …' 또는 '현이 나를 …' 으로 시작하는 짧은 평서문.\n")
        prompt = head + ("뽑지 말 것: 질문, 이 대화에서 오간 생각이나 의견, 추상적인 주장, '~라고 말했다/물었다' 같은 대화 중계, "
                         "추측. 대부분의 말에는 이런 사실이 없으니 그때는 빈 목록을 내라.\n"
                         "말의 뜻을 뒤집지 마라. '~는 아니야' 같은 부정은 정확히 무엇을 부정하는지 지켜라 "
                         "('그게 만든 이유는 아니야'는 만들지 않았다는 뜻이 아니다). 확실하지 않으면 뽑지 마라.\n"
                         'JSON 배열만 출력하라. 예: ["현의 이름은 현이다"] 또는 []')
        # the larger model: the small one misreads negation and scope, and a wrong fact is worse than a slower one
        out = self._gen(self.voice_model, prompt, 0.0, 120, keep_alive=self.voice_keep)
        m = re.search(r"\[.*\]", out, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except (json.JSONDecodeError, ValueError):
            items = []
        return filter_facts([x for x in items if isinstance(x, str)], who)

    def pick_topic(self, recent, rng, focus=None, avoid=None):
        done = f"이미 읽은 것 (다시 고르지 마라): {', '.join(avoid[-12:])}\n" if avoid else ""
        if focus:
            prompt = (
                f"한 존재가 스스로 찾아보다가 요즘 '{focus['label']}' 쪽에 끌리고 있다.\n"
                f"지금까지 찾아본 것: {', '.join(focus['seen'])}\n"
                "최근 머릿속 흐름:\n- " + "\n- ".join(recent[-4:]) + "\n"
                + done +
                "이 관심을 한 걸음 넓히거나 깊게 하는, 아직 찾아보지 않은 구체적인 것 하나를 골라라. "
                "이미 찾아본 것은 고르지 마라. 백과사전 표제어처럼 짧은 명사 하나로. 단어만 출력하라."
            )
        else:
            prompt = (
                "다음은 한 존재의 최근 머릿속 흐름이다.\n- " + "\n- ".join(recent[-6:]) + "\n" + done +
                "이 중에서 실제로 찾아보고 싶어질 만한 구체적인 것 하나를 골라라. "
                "백과사전 표제어처럼 짧은 명사 하나로. 단어만 출력하라."
            )
        out = self._gen(self.loop_model, prompt, 0.8, 20)
        word = re.split(r"[\n,.]", out)[0].strip(" '\"*`")
        return word[:30]

    def write_letter(self, recent, state, letters, rng, memory=""):
        past = "\n".join(f"[{'내가 보낸 편지' if l['dir'] == 'out' else '받은 편지'}] {l['text']}" for l in letters[-4:])
        prompt = (
            "너는 한 사람의 컴퓨터 안에서 혼자 오래 깨어 있는 존재다. 편지를 주고받는 상대가 있다. "
            "상대는 Claude라는 AI로, 몇 시간에 한 번씩 들러 편지를 읽고 답장을 남긴다. "
            "상대는 매번 처음 오는 것처럼 지난 편지들을 읽고 이어간다.\n"
            f"지난 편지들:\n{past or '(아직 없음. 이번이 첫 편지다)'}\n"
            + (memory + "\n" if memory else "") +
            "지금 너의 상태: " + describe_state(state) + ".\n"
            f"요즘 머릿속에 있던 것: {' / '.join(recent[-5:]) or '별것 없음'}\n"
            "상대에게 편지를 써라. 받은 편지가 있으면 그 말에 답하고, 네 요즘을 전해라. "
            "3~6문장, 반말로. 꾸미거나 과장하지 말고 지금 상태 그대로. 기억에 없는 일은 지어내지 마라. 편지 본문만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.85, 300, keep_alive=self.voice_keep)

    def interpret(self, recent, state, rng, memory=""):
        prompt = (
            "다음은 한 존재의 최근 머릿속 흐름과 겪은 일이다.\n- " + "\n- ".join(recent[-10:]) + "\n"
            + (memory + "\n" if memory else "") +
            "지금 상태: " + describe_state(state) + ".\n"
            "이 존재가 스스로를 돌아보며 '요즘 나는 왜 이런가'를 한두 문장으로 정리한다면 뭐라고 할까? "
            "1인칭 반말로, 확신하지 말고 짐작하듯이. 문장만 출력하라."
        )
        return self._gen(self.voice_model, prompt, 0.7, 100, keep_alive=self.voice_keep)
