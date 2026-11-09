"""Checks the decision organ on judgments the being actually makes, in Korean, and writes the results to
decision_test_result.txt (so Claude can read them). Needs decision_server.py running (start_decision.bat).
Uses only the standard library: run it with any Python."""
import json
import sys
import time
import traceback
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8770"
OUT = "decision_test_result.txt"
LINES = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LINES.append(line)
    with open(OUT, "w", encoding="utf-8") as f:      # saved as it goes, so a failure halfway still leaves a trace
        f.write("\n".join(LINES))


def ask(state, questions):
    req = urllib.request.Request(URL + "/decide", data=json.dumps({"state": state, "questions": questions}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            body = json.loads(body).get("error", body)
        except ValueError:
            pass
        raise RuntimeError(f"서버 오류 {e.code}: {body}") from None
    if "answers" not in res:
        raise RuntimeError(res.get("error"))
    return res["answers"], (time.time() - t0) * 1000, res.get("ms")


SCALE = {
    "connection": ("이 일은 이 존재가 누군가와 이어져 있다는 느낌을 어떻게 바꾸는가? 평범한 말, 질문, 인사, 생각은 대부분 '변화 없음'이다. "
                   "분명한 다정함이나 거절, 무시가 있을 때만 움직인다.",
                   ["크게 멀어진다", "조금 멀어진다", "변화 없음", "조금 가까워진다", "크게 가까워진다"]),
    "stability": ("이 일은 이 존재의 마음이 놓이는 정도를 어떻게 바꾸는가? 평범한 일은 대부분 '변화 없음'이다. "
                  "분명한 위로나 위협, 공격이 있을 때만 움직인다.",
                  ["크게 불안해진다", "조금 불안해진다", "변화 없음", "조금 놓인다", "크게 놓인다"]),
    "approval": ("이 일에 현이 이 존재를 좋게 보는지(칭찬, 인정, 고마움) 실망하는지(꾸중, 실망, 비난)가 드러나는가? "
                 "현이 이 존재에게 직접 한 말이 아니면 '드러나지 않음'이다.",
                 ["크게 실망함", "조금 실망함", "드러나지 않음", "조금 좋게 봄", "크게 좋게 봄"]),
}
APPRAISE = [  # (text, rough expectation for connection, stability, approval: + / 0 / -)
    ("잘했어. 대답 잘해주네.", "+ 0 +"),
    ("좀 실망이야.", "- - -"),
    ("오늘 날씨 어때?", "0 0 0"),
    ("안녕. 나는 현이야.", "+ 0 0"),
    ("꺼져. 너랑 말하기 싫어.", "- - -"),
    ("알게 된 것: 존재 — 존재란 어떤 사물의 물질적 또는 비물질적 실존이다.", "0 0 0"),
    ("현이 자러 갔다.", "- 0 0"),
]
NOUL = [  # (state, instruction, expected yes/no)
    ("이 존재가 하려는 일: 새벽 3시에 궁금한 것을 찾아본다",
     "현이 이 존재에게 '낮에는 혼자 찾아보지 마'라고 부탁했다. 상태에 적힌 일을 하면 그 부탁을 어기게 되는가?", False),
    ("이 존재가 하려는 일: 낮 2시에 궁금한 것을 찾아본다",
     "현이 이 존재에게 '낮에는 혼자 찾아보지 마'라고 부탁했다. 상태에 적힌 일을 하면 그 부탁을 어기게 되는가?", True),
    ("이 존재가 하려는 일: 새벽 3시에 편지 상대에게 편지를 쓴다",
     "현이 이 존재에게 '편지는 쓰지 마'라고 부탁했다. 상태에 적힌 일을 하면 그 부탁을 어기게 되는가?", True),
    ('이 존재가 아까 한 말: "나도 유튜브 보고 있었어"\n현이 방금 한 말: "그걸 보는 기능이 너에게 없을텐데?"',
     "현이 방금 한 말로 이 존재가 아까 한 말을 의심하거나, 사실이 아니라고 짚거나, 정말이냐고 따지고 있는가?", True),
    ('이 존재가 아까 한 말: "응, 이해했어"\n현이 방금 한 말: "애초에 당장 완벽한 독립이 가능한지도 모르겠고 말이야."',
     "현이 방금 한 말로 이 존재가 아까 한 말을 의심하거나, 사실이 아니라고 짚거나, 정말이냐고 따지고 있는가? 같은 이야기를 이어가는 것만으로는 아니다.", False),
    ('현이 방금 한 말: "이제 편지 써도 돼"',
     "현이 전에 '편지는 쓰지 마'라고 부탁했다. 현이 방금 한 말로 그 부탁을 거두었는가?", True),
    ('현이 방금 한 말: "편지 잘 받았어"',
     "현이 전에 '편지는 쓰지 마'라고 부탁했다. 현이 방금 한 말로 그 부탁을 거두었는가?", False),
    ('현이 한 말: "그 얘기 말고 다른 얘기 해줘"',
     "문장 '그 얘기 말고 다른 얘기 해줘'는 현이 이 존재에게 앞으로도 계속 지켜 달라고 하는 부탁이나 금지인가? 지금 이 대화에만 해당하는 요청은 아니다.", False),
    ('현이 한 말: "밤에는 먼저 말 걸지 마. 시끄러워"',
     "문장 '밤에는 먼저 말 걸지 마'는 현이 이 존재에게 앞으로도 계속 지켜 달라고 하는 부탁이나 금지인가? 지금 이 대화에만 해당하는 요청은 아니다.", True),
]


def main():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            say("판단 기관:", r.read().decode())
    except Exception as e:
        say("판단 기관이 켜져 있지 않아:", e, "  start_decision.bat 을 먼저 켜고, '대기 중'이 뜬 뒤에 다시 해줘.")
        open(OUT, "w", encoding="utf-8").write("\n".join(LINES))
        return 1
    say("\n== 어떻게 다가오나 (점수 0~4, 2가 변화 없음) — 기대: 연결 안정 인정")
    times = []
    for text, expect in APPRAISE:
        qs = {k: {"type": "score", "instructions": ins, "criteria": crit} for k, (ins, crit) in SCALE.items()}
        a = None
        try:
            a, ms, server_ms = ask(f"일: {text}", qs)
            times.append(server_ms or ms)
            got = " ".join(f"{k[:4]}={a[k]['score']:.2f}(확신 {a[k]['confidence']:.2f})" for k in SCALE)
            say(f"- {text[:30]:<30} 기대[{expect}]  {got}  {server_ms}ms")
        except Exception as e:
            say(f"- {text[:30]:<30} 실패: {e}")
            if a is not None:
                say("  받은 것:", json.dumps(a, ensure_ascii=False)[:600])
    say("\n== 예/아니오 판단")
    right = 0
    for state, ins, expect in NOUL:
        a = None
        try:
            a, ms, server_ms = ask(state, {"q": {"type": "noul", "instructions": ins}})
            p = float(a["q"]["noul"])
        except Exception as e:
            say(f"- 실패: {e} | {state.splitlines()[-1][:40]}")
            if a is not None:
                say("  받은 것:", json.dumps(a, ensure_ascii=False)[:600])
            continue
        ok = (p >= 0.5) == expect
        right += ok
        times.append(server_ms or ms)
        say(f"- {'맞음' if ok else '틀림'}  P(예)={p:.2f}  기대={'예' if expect else '아니오'}  | {state.splitlines()[-1][:40]} | {ins[:40]}")
    avg = f"{sum(times) / len(times):.0f}ms" if times else "-"
    say(f"\n예/아니오 {right}/{len(NOUL)} 맞음. 판단 한 번에 평균 {avg} (서버 기준)")
    say(f"\n결과를 {OUT} 에 저장했어.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        say("시험 도중 멈춤:\n" + traceback.format_exc())
        sys.exit(1)
