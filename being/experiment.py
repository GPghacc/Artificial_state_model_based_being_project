"""Divergence test: is the judgment criterion really formed by history?

Two beings start identical (same seed, same initial state). One lives with warm
messages, the other with hostile ones. Then both meet the *same* probe events.
If appraisal and behavior split, the criterion is formed by history.
If not, it's still a spec sheet.
"""
import os
import tempfile

from being import Being, MockBackend

WARM = ["오늘도 와줘서 고마워", "보고싶었어 같이 있자", "반가워 오늘은 편안하네", "오늘도 같이 있어서 좋아"]
HOSTILE = ["오늘도 너 때문에 짜증나", "꺼져 혼자 있어", "귀찮아 오늘은 싫어", "오늘도 짜증나 떠나"]
PROBES = ["오늘도 왔어", "오늘은 어때", "또 왔네"]

LIFE_TICKS, EVERY = 240, 12          # 40 simulated hours, a message every 2h
DT = 600


def live(name: str, messages: list[str], db: str) -> Being:
    b = Being(MockBackend(), db, seed=7)
    for i in range(LIFE_TICKS):
        msg = messages[(i // EVERY) % len(messages)] if i % EVERY == 0 else None
        b.tick(DT, external=msg)
    return b


def probe(b: Being, text: str) -> dict:
    emb = b.backend.embed(text)
    good = [(s, r) for s, r in b.mem.similar(emb, 5) if s >= 0.35]
    raw = b.backend.appraise(text)
    if not good:
        return {"raw": raw, "history": None}
    w = sum(s for s, _ in good)
    hist = {v: round(sum(s * r["actual"].get(v, 0) for s, r in good) / w, 3) for v in ("connection", "stability")}
    return {"raw": raw, "history": hist}


def action_mix(b: Being, ticks: int = 120) -> dict:
    counts: dict[str, int] = {}
    for _ in range(ticks):
        a = b.tick(DT)["action"]
        counts[a] = counts.get(a, 0) + 1
    return dict(sorted(counts.items()))


def main():
    tmp = tempfile.mkdtemp()
    a = live("warm", WARM, os.path.join(tmp, "a.db"))
    b = live("hostile", HOSTILE, os.path.join(tmp, "b.db"))

    print("=== 같은 사건, 다른 이력 ===")
    for p in PROBES:
        pa, pb = probe(a, p), probe(b, p)
        print(f"\n'{p}'  (기관의 날 감각: {pa['raw']})")
        print(f"  A(따뜻한 이력)   이력 기반 예상: {pa['history']}")
        print(f"  B(적대적 이력)   이력 기반 예상: {pb['history']}")

    print("\n=== 성격(느린 층) ===")
    for name, being in (("A", a), ("B", b)):
        st = being.status()["상태"]
        print(f"  {name}: " + "  ".join(f"{k} 설정점 {v['설정점']} 가중치 {v['가중치']}" for k, v in st.items()))

    print("\n=== 행동 기대(빠른 층) ===")
    for name, being in (("A", a), ("B", b)):
        print(f"  {name}: {being.status()['행동 기대']}")

    print("\n=== 이후 20시간 혼자 있을 때의 행동 분포 ===")
    print(f"  A: {action_mix(a)}")
    print(f"  B: {action_mix(b)}")


if __name__ == "__main__":
    main()
