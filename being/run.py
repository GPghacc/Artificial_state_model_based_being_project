"""Run the being.

Simulated time (fast):
    python run.py --ticks 300 --sim-minutes 10
    python run.py --ticks 300 --inject "40:오늘도 와줘서 고마워" --inject "120:꺼져"

Real time (type messages while it runs; /status, /surfaced, /quit):
    python run.py --realtime --tick-seconds 5 --time-scale 60

Ollama:
    python run.py --backend ollama --model llama3.1:8b --embed-model nomic-embed-text ...
"""
import argparse
import json
import queue
import threading
import time

from being import Being, MockBackend, OllamaBackend

ICON = {"rest": "·", "wander": "~", "recall": "↺", "reach_out": "!", "attend": ">"}


def line(i: int, rec: dict, being: Being) -> str:
    s = being.state.values()
    head = f"[{i:4d}] {ICON.get(rec['action'], '?')} {rec['action']:<9} " \
           f"C{s['connection']:.2f} N{s['novelty']:.2f} S{s['stability']:.2f} 불편{rec['discomfort']:.3f}"
    if rec["action"] == "rest":
        return head
    text = rec["text"] if len(rec["text"]) < 70 else rec["text"][:67] + "…"
    out = f"{head}  {text}"
    if rec["surfaced"]:
        out += f"\n       ⤴ 떠오름({rec['surfaced']})"
    return out


def make_backend(a):
    if a.backend == "ollama":
        return OllamaBackend(a.model, a.embed_model, a.host)
    return MockBackend()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--db", default="being.db")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--sim-minutes", type=float, default=10, help="simulated minutes per tick")
    p.add_argument("--inject", action="append", default=[], help="TICK:message")
    p.add_argument("--quiet", action="store_true", help="hide rest ticks")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--tick-seconds", type=float, default=5)
    p.add_argument("--time-scale", type=float, default=60, help="sim seconds per real second")
    a = p.parse_args()

    being = Being(make_backend(a), a.db, seed=a.seed)

    def flush_outbox():
        while being.outbox:
            print(f"       💬 {being.outbox.pop(0)}")

    if not a.realtime:
        injects = {}
        for item in a.inject:
            t, msg = item.split(":", 1)
            injects[int(t)] = msg
        for i in range(a.ticks):
            rec = being.tick(a.sim_minutes * 60, external=injects.get(i))
            if not (a.quiet and rec["action"] == "rest"):
                print(line(i, rec, being))
            flush_outbox()
        print("\n" + json.dumps(being.status(), ensure_ascii=False, indent=2))
        return

    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [q.put(l.rstrip("\n")) for l in iter(input, None)], daemon=True).start()
    print("실시간 모드. 메시지를 입력하면 다음 틱에 들어갑니다. /status /surfaced /quit")
    i, last = 0, time.time()
    while True:
        time.sleep(a.tick_seconds)
        msg = None
        while not q.empty():
            cmd = q.get()
            if cmd == "/quit":
                return
            if cmd == "/status":
                print(json.dumps(being.status(), ensure_ascii=False, indent=2))
            elif cmd == "/surfaced":
                for t, reason, text in being.mem.surfaced(10):
                    print(f"  {t/3600:7.2f}h [{reason}] {text}")
            elif cmd:
                msg = cmd
        now = time.time()
        rec = being.tick((now - last) * a.time_scale, external=msg)
        last = now
        if not (a.quiet and rec["action"] == "rest"):
            print(line(i, rec, being))
        flush_outbox()
        i += 1


if __name__ == "__main__":
    main()
