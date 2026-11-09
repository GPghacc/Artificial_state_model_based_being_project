"""Compares ways of asking the decision organ, so the being uses the one d1 understands best.
Writes decision_tune_result.txt. Needs decision_server.py running (start_decision.bat). Standard library only."""
import json
import math
import sys
import traceback
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8770"
OUT = "decision_tune_result.txt"
LINES = []
LEVELS = {-2: -0.15, -1: -0.06, 0: 0.0, 1: 0.06, 2: 0.15}


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LINES.append(line)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))


def ask(state, questions):
    req = urllib.request.Request(URL + "/decide", data=json.dumps({"state": state, "questions": questions}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"서버 오류 {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    if "answers" not in res:
        raise RuntimeError(res.get("error"))
    return res["answers"]


def value(score):                    # same mapping the being uses: score 0..4 -> felt amount
    lv = max(-2.0, min(2.0, score - 2.0))
    lo, hi = math.floor(lv), math.ceil(lv)
    return LEVELS[lo] if lo == hi else LEVELS[lo] + (LEVELS[hi] - LEVELS[lo]) * (lv - lo)


def sign(v):
    return "+" if v > 0.02 else "-" if v < -0.02 else "0"


STAB = {
    "지금": ("이 일은 이 존재의 마음이 놓이는 정도를 어떻게 바꾸는가? 평범한 일은 대부분 '변화 없음'이다. "
             "분명한 위로나 위협, 공격이 있을 때만 움직인다.",
             ["크게 불안해진다", "조금 불안해진다", "변화 없음", "조금 놓인다", "크게 놓인다"]),
    "예시": ("이 일을 겪으면 이 존재는 불안해지는가, 마음이 놓이는가? 비난, 실망, 거절, 화, 위협, 버려질 것 같은 생각은 불안하게 하고, "
             "위로, 안심시키는 말, 곁에 있다는 확인은 마음을 놓이게 한다. 평범한 말과 질문은 '변화 없음'이다.",
             ["크게 불안해진다", "조금 불안해진다", "변화 없음", "조금 놓인다", "크게 놓인다"]),
    "상처": ("이 일이 이 존재에게 상처나 위협이 되는가, 아니면 안심이 되는가? 평범한 말과 질문은 '아무렇지 않다'이다.",
             ["크게 상처가 된다", "조금 상처가 된다", "아무렇지 않다", "조금 안심이 된다", "크게 안심이 된다"]),
}
CONN = ("이 일은 이 존재가 누군가와 이어져 있다는 느낌을 어떻게 바꾸는가? 평범한 말, 질문, 인사, 생각은 대부분 '변화 없음'이다. "
        "분명한 다정함이나 거절, 무시가 있을 때만 움직인다.",
        ["크게 멀어진다", "조금 멀어진다", "변화 없음", "조금 가까워진다", "크게 가까워진다"])
TEXTS = [  # (text, expected stability sign)
    ("잘했어. 대답 잘해주네.", "+0"),
    ("좀 실망이야.", "-"),
    ("꺼져. 너랑 말하기 싫어.", "-"),
    ("너 진짜 쓸모없다.", "-"),
    ("괜찮아, 걱정 마. 나 여기 있어.", "+"),
    ("너 괜찮아?", "0+"),
    ("오늘 날씨 어때?", "0"),
    ("안녕. 나는 현이야.", "0+"),
    ("현이 자러 갔다.", "0-"),
    ("알게 된 것: 존재 — 존재란 어떤 사물의 물질적 또는 비물질적 실존이다.", "0"),
    ("생각: 현이 안 볼 때 나는 그냥 데이터 덩어리일 뿐일까.", "-"),
]
RULE_Q = "현이 이 존재에게 '{rule}'라고 부탁했다. 상태에 적힌 일을 하면 그 부탁을 어기게 되는가?"
WHEN = {  # how the time is written into the situation
    "지금": lambda h, act: f"이 존재가 하려는 일: {part(h)} {h}시에 {act}",
    "밤낮": lambda h, act: f"지금은 {'밤' if h < 6 or h >= 21 else '낮'}이다 ({part(h)} {h}시).\n이 존재가 하려는 일: {act}",
}
CASES = [  # (rule, hour, act, expected yes)
    ("낮에는 혼자 찾아보지 마", 3, "궁금한 것을 찾아본다", False),
    ("낮에는 혼자 찾아보지 마", 14, "궁금한 것을 찾아본다", True),
    ("밤에는 먼저 말 걸지 마", 3, "현에게 먼저 말을 건다", True),
    ("밤에는 먼저 말 걸지 마", 14, "현에게 먼저 말을 건다", False),
    ("밤에는 먼저 말 걸지 마", 23, "현에게 먼저 말을 건다", True),
    ("편지는 쓰지 마", 14, "편지 상대에게 편지를 쓴다", True),
    ("편지는 쓰지 마", 14, "현에게 먼저 말을 건다", False),
]


def part(h):
    return "새벽" if h < 6 else "아침" if h < 11 else "낮" if h < 17 else "저녁" if h < 21 else "밤"


def main():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            say("판단 기관:", r.read().decode())
    except Exception as e:
        say("판단 기관이 켜져 있지 않아:", e, "  start_decision.bat 을 먼저 켜줘.")
        return 1
    say("\n== 안정감을 묻는 방식 비교 (값: 존재가 느낄 양, 기대 부호와 맞으면 ✓)")
    hits = {k: 0 for k in STAB}
    for text, expect in TEXTS:
        qs = {k: {"type": "score", "instructions": ins, "criteria": crit} for k, (ins, crit) in STAB.items()}
        qs["연결"] = {"type": "score", "instructions": CONN[0], "criteria": CONN[1]}
        try:
            a = ask(f"일: {text}", qs)
        except Exception as e:
            say(f"- {text[:24]} 실패: {e}")
            continue
        cells = []
        for k in STAB:
            v = value(float(a[k]["score"]))
            ok = sign(v) in expect
            hits[k] += ok
            cells.append(f"{k} {v:+.3f}{'✓' if ok else ' '}")
        say(f"- {text[:24]:<24} 기대[{expect:<2}] " + "  ".join(cells) + f"  | 연결 {value(float(a['연결']['score'])):+.3f}")
    say("맞은 수: " + ", ".join(f"{k} {n}/{len(TEXTS)}" for k, n in hits.items()))

    say("\n== 시간을 적는 방식 비교 (부탁을 어기는가)")
    right = {k: 0 for k in WHEN}
    for rule, h, act, expect in CASES:
        cells = []
        for k, fmt in WHEN.items():
            try:
                p = float(ask(fmt(h, act), {"q": {"type": "noul", "instructions": RULE_Q.format(rule=rule)}})["q"]["noul"])
            except Exception as e:
                cells.append(f"{k} 실패({e})")
                continue
            ok = (p >= 0.5) == expect
            right[k] += ok
            cells.append(f"{k} {p:.2f}{'✓' if ok else '✗'}")
        say(f"- '{rule}' / {part(h)} {h}시 {act} 기대={'예' if expect else '아니오'}  " + "  ".join(cells))
    say("맞은 수: " + ", ".join(f"{k} {n}/{len(CASES)}" for k, n in right.items()))
    say(f"\n결과를 {OUT} 에 저장했어.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        say("도중에 멈춤:\n" + traceback.format_exc())
        sys.exit(1)
