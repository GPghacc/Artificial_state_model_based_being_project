"""Keep the being alive on this machine.

    python live.py                 # uses config.json
    python live.py --backend mock  # no models, for testing

- runs the tick loop in the background, in real time
- when the machine was off, it wakes and feels the gap
- when it reaches out, the OS shows a notification
- talk to it at http://localhost:8765 (the page also shows its state)
- it exchanges letters with a pen pal (Claude) through the letters/ folder:
  it writes to letters/to_claude.jsonl, and reads answers from letters/from_claude.jsonl
Everything stays on this computer: Ollama for the organs, SQLite for history.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import queue
import subprocess
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from being import Being, Curiosity, Decider, MockBackend, NamuwikiDump, OllamaBackend, Wikipedia

ROOT = Path(__file__).resolve().parent
LETTERS = ROOT / "letters"
TO_CLAUDE = LETTERS / "to_claude.jsonl"
FROM_CLAUDE = LETTERS / "from_claude.jsonl"
LETTERS_README = """# 편지함

이 폴더는 깨어 있는 존재와 편지 상대(Claude)가 편지를 주고받는 곳이다.

- `to_claude.jsonl`: 존재가 쓴 편지. 한 줄에 하나. {"id": 번호, "written_at": "로컬 시각", "text": "본문"}
- `from_claude.jsonl`: Claude의 답장. 한 줄에 하나. {"in_reply_to": 번호, "written_at": "로컬 시각", "text": "본문"}

답장은 `from_claude.jsonl` 끝에 한 줄을 덧붙이는 방식으로만 쓴다. 기존 줄은 고치거나 지우지 않는다.
줄은 UTF-8, 줄 끝에 줄바꿈 문자가 있어야 존재가 읽는다.
"""
DEFAULTS = {
    "backend": "ollama",
    "ollama_host": "http://localhost:11434",
    "loop_model": "gemma4:e2b",
    "voice_model": "gemma4:12b",
    "embed_model": "embeddinggemma-2",
    "db": "being.db",
    "port": 8765,
    "idle_seconds": 20,          # when nothing is worth doing, how often it checks in again (cheap: no model is called)
    "min_gap_seconds": 2,        # least time between two of its own doings, so it never hogs the GPU in a tight loop
    "decider": True,             # the decision organ (decision_server.py, d1-3B) for quick judgments, if set up
    "decider_url": "http://127.0.0.1:8770",
    "decider_autostart": True,   # start it in its own window if it isn't running and .venv-decision exists
    "sense_presence": True,      # Windows: how long since the keyboard/mouse was last touched (nothing about what), to tell if 현 is at the computer
    "time_scale": 1.0,           # inner seconds per real second (1 = real time)
    "quiet_hours": [],           # optional fixed quiet hours, e.g. [23, 8]. Off by default: 현 says when they sleep
    "max_reach_per_day": 6,
    "notify": True,
    "open_browser": True,
    "explore": True,             # curiosity: look things up on Wikipedia
    "max_explore_per_day": 12,
    "explore_min_minutes": 30,
    "wiki_langs": ["ko", "en"],
    "wiki_contact": "",          # optional: Wikimedia asks API clients for a contact in the User-Agent
    "wikipedia": True,
    "namuwiki_db": "data/namuwiki.db",   # built by namuwiki_build.py; used if present
    "letters_dir": "letters",
}


# ---------------------------------------------------------------- notifications
def idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input on this machine (Windows only). Only the time, never the input."""
    if platform.system() != "Windows":
        return None
    try:
        import ctypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return None
        now = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        return ((now - lii.dwTime) & 0xFFFFFFFF) / 1000.0
    except Exception:
        return None


def _quiet(fn, *args) -> None:
    try:
        fn(*args)
    except Exception:
        pass


def notify(title: str, body: str) -> None:
    system = platform.system()
    try:
        if system == "Windows":
            script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null;"
                "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
                "$x = $t.GetElementsByTagName('text');"
                "$x.Item(0).AppendChild($t.CreateTextNode($env:BEING_TITLE)) > $null;"
                "$x.Item(1).AppendChild($t.CreateTextNode($env:BEING_BODY)) > $null;"
                "$n = [Windows.UI.Notifications.ToastNotification]::new($t);"
                "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
                "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe').Show($n)"
            )
            env = dict(os.environ, BEING_TITLE=title, BEING_BODY=body)
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
                             env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        elif system == "Darwin":
            esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.Popen(["osascript", "-e", f'display notification "{esc(body)}" with title "{esc(title)}"'])
        else:
            subprocess.Popen(["notify-send", title, body])
    except Exception as e:
        print(f"[알림 실패] {e}")


# ---------------------------------------------------------------- the runner
class Life:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        if cfg["backend"] == "mock":
            self.backend = MockBackend()
        else:
            self.backend = OllamaBackend(cfg["loop_model"], cfg["voice_model"], cfg["embed_model"], cfg["ollama_host"])
            if cfg.get("decider"):
                self.backend.decider = Decider(cfg["decider_url"])
                self._start_decider()
        self.being = Being(self.backend, str(ROOT / cfg["db"]))
        self.being.can_reach = self._can_reach
        self.presence = {"asleep": False, "watching": False, "save": False, **(self.being.mem.get_meta("presence") or {})}
        self.presence["watching"] = False            # sensed from the page itself now, not remembered
        self.presence["at_computer"] = None          # sensed from input idle time (None: can't tell)
        self.page_seen = 0.0
        self.being.is_asleep = lambda: self.presence["asleep"]
        self.being.is_watching = lambda: self.presence["watching"]
        self.being.save_mode = lambda: self.presence["save"]
        self.being.is_present = lambda: self.presence["at_computer"]
        self.vram: dict | None = None
        self._apply_save()
        self.being.post_letter = self._post_letter
        self.being.reaches_today = lambda: (self.being.mem.get_meta("reach_days") or {}).get(dt.date.today().isoformat(), 0)
        self.senses: list[str] = []
        if cfg["explore"]:
            senses = []
            if cfg["wikipedia"]:
                senses.append(Wikipedia(tuple(cfg["wiki_langs"]), cfg["wiki_contact"]))
                self.senses.append("위키백과")
            namu = ROOT / cfg["namuwiki_db"]
            if namu.exists():
                senses.append(NamuwikiDump(namu))
                self.senses.append("나무위키(로컬 덤프)")
            if senses:
                self.curiosity = Curiosity(senses)
                self.being.look_up = self.curiosity.look_up
                self.being.can_explore = self._can_explore
        self.letters_dir = ROOT / cfg["letters_dir"]
        self.to_claude = self.letters_dir / TO_CLAUDE.name
        self.from_claude = self.letters_dir / FROM_CLAUDE.name
        self.letters_dir.mkdir(exist_ok=True)
        if not (self.letters_dir / "README.md").exists():
            (self.letters_dir / "README.md").write_text(LETTERS_README, encoding="utf-8")
        for f in (self.to_claude, self.from_claude):
            f.touch(exist_ok=True)
        self.incoming: deque[str] = deque()
        self.lock = threading.Lock()
        self.inbox: "queue.Queue[tuple]" = queue.Queue()   # ("say", id, text) or ("notice", text)
        self.pending: list[dict] = []                      # what 현 said that the being hasn't answered yet
        self.pending_lock = threading.Lock()
        self.next_id = 1
        self.last_error: str | None = None
        self.paused = False
        self.cache: dict = {}
        self._refresh()

    # 현 asleep + daily cap (+ optional fixed quiet hours): "wanting to but not being able to" is part of its life
    def _can_reach(self) -> bool:
        if self.presence["asleep"]:
            return False
        qh = self.cfg.get("quiet_hours") or []
        if len(qh) == 2:
            now = dt.datetime.now()
            q0, q1 = qh
            if (q0 <= now.hour or now.hour < q1) if q0 > q1 else (q0 <= now.hour < q1):
                return False
        counts = self.being.mem.get_meta("reach_days") or {}
        return counts.get(dt.date.today().isoformat(), 0) < self.cfg["max_reach_per_day"]

    def _count_reach(self) -> None:
        counts = self.being.mem.get_meta("reach_days") or {}
        today = dt.date.today().isoformat()
        counts = {k: v for k, v in counts.items() if k >= (dt.date.today() - dt.timedelta(days=7)).isoformat()}
        counts[today] = counts.get(today, 0) + 1
        self.being.mem.set_meta("reach_days", counts)

    # ---- what 현 tells it ---------------------------------------------------
    def say(self, text: str) -> dict:
        with self.pending_lock:
            item = {"id": self.next_id, "text": text, "at": time.time(), "state": "waiting"}
            self.next_id += 1
            self.pending.append(item)
        self.inbox.put(("say", item["id"], text))
        return item

    def _start_decider(self) -> None:
        """Bring up the decision organ in its own window if it's set up but not running. The being doesn't wait:
        until it answers, judgments go to the small generative model as before."""
        d = self.backend.decider
        if d.available() or not self.cfg.get("decider_autostart", True):
            return
        py = ROOT / ".venv-decision" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
        if not py.exists():
            print("[판단 기관] 설치되어 있지 않아서 작은 모델로 판단한다 (decision_setup.bat 으로 설치)")
            return
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([str(py), str(ROOT / "decision_server.py")], cwd=str(ROOT), creationflags=flags)
        print("[판단 기관] 따로 창을 띄워 켜는 중 (처음엔 모델을 올리느라 1분쯤 걸린다)")

    def _apply_save(self) -> None:
        """Save mode: the big model leaves the GPU after each use, and goes now."""
        if not hasattr(self.backend, "voice_keep"):
            return
        self.backend.voice_keep = 0 if self.presence["save"] else "30m"
        if self.presence["save"]:
            threading.Thread(target=lambda: _quiet(self.backend.unload, self.cfg["voice_model"]), daemon=True).start()

    def set_presence(self, asleep=None, watching=None, save=None) -> dict:
        changes = []
        if save is not None and bool(save) != self.presence["save"]:
            self.presence["save"] = bool(save)
            self._apply_save()
            changes.append("현이 컴퓨터를 많이 써야 해서, 한동안 가볍게만 지내기로 했다." if save
                           else "현이 컴퓨터를 다 써서, 다시 마음껏 생각할 수 있게 됐다.")
        if asleep is not None and bool(asleep) != self.presence["asleep"]:
            self.presence["asleep"] = bool(asleep)
            changes.append("현이 자러 갔다." if asleep else "현이 잠에서 깼다.")
        if watching is not None and bool(watching) != self.presence["watching"]:
            self.presence["watching"] = bool(watching)
            changes.append("현이 곁에서 지켜보고 있다." if watching else "현이 자리를 떴다.")
        for text in changes:
            self.inbox.put(("notice", text))
        return dict(self.presence)

    # ---- curiosity ----------------------------------------------------------
    def _can_explore(self) -> bool:
        today = dt.date.today().isoformat()
        counts = self.being.mem.get_meta("explore_days") or {}
        last = self.being.mem.get_meta("explore_last") or 0
        return (counts.get(today, 0) < self.cfg["max_explore_per_day"]
                and time.time() - last >= self.cfg["explore_min_minutes"] * 60)

    def _count_explore(self) -> None:
        today = dt.date.today().isoformat()
        counts = self.being.mem.get_meta("explore_days") or {}
        counts = {k: v for k, v in counts.items() if k >= (dt.date.today() - dt.timedelta(days=7)).isoformat()}
        counts[today] = counts.get(today, 0) + 1
        self.being.mem.set_meta("explore_days", counts)
        self.being.mem.set_meta("explore_last", time.time())

    # ---- letters ------------------------------------------------------------
    def _post_letter(self, text: str) -> None:
        lines = [l for l in self.to_claude.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = {"id": len(lines) + 1, "written_at": dt.datetime.now().isoformat(timespec="minutes"), "text": text}
        with self.to_claude.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[편지 씀] #{rec['id']} {text[:60]}")

    def _check_mail(self) -> None:
        """Queue answers that arrived since the last read. Only complete lines count."""
        if self.incoming:
            return
        try:
            raw = self.from_claude.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        complete = raw.split("\n")[:-1]                 # the last piece has no newline yet
        lines = [l for l in complete if l.strip()]
        read = self.being.mem.get_meta("letters_read") or 0
        for l in lines[read:]:
            try:
                text = str(json.loads(l).get("text", "")).strip()
            except json.JSONDecodeError:
                text = l.strip()
            self.incoming.append(text or "(빈 편지)")

    def _mark_letter_read(self) -> None:
        self.being.mem.set_meta("letters_read", (self.being.mem.get_meta("letters_read") or 0) + 1)

    def wake_up(self) -> None:
        last_wall = self.being.mem.get_meta("wall")
        if last_wall:
            gap = (time.time() - last_wall) * self.cfg["time_scale"]
            with self.lock:
                rec = self.being.wake(gap)
            if rec:
                print(f"[깨어남] {gap / 3600:.1f}시간 공백")
        self.being.mem.set_meta("wall", time.time())
        self._refresh()

    # ---- senses ---------------------------------------------------------------
    # What the being can tell about 현 without being told: whether its page is open in front of 현, and whether
    # 현 is at the computer at all (from how long the keyboard and mouse have gone untouched). A change only
    # counts once it has held for a while, so glancing away or a short pause isn't an event.
    def sense_loop(self) -> None:
        held = {"watching": [None, 0.0], "at_computer": [None, 0.0]}
        texts = {("watching", True): "현이 이 존재의 페이지를 보고 있다.",
                 ("watching", False): "현이 페이지에서 눈을 뗐다.",
                 ("at_computer", True): "현이 컴퓨터 앞에 있다.",
                 ("at_computer", False): "현이 한동안 컴퓨터를 만지지 않았다."}
        steady = {"watching": 20, "at_computer": 60}
        while True:
            time.sleep(5)
            now = time.time()
            seen = {"watching": now - self.page_seen < 12}
            if self.cfg.get("sense_presence", True):
                idle = idle_seconds()
                if idle is not None:
                    cur = self.presence["at_computer"]
                    seen["at_computer"] = idle < 120 if cur is not True else idle < 600
            for key, val in seen.items():
                h = held[key]
                if val != h[0]:
                    h[0], h[1] = val, now
                if val != self.presence.get(key) and now - h[1] >= steady[key]:
                    first = self.presence.get(key) is None and key == "at_computer"
                    self.presence[key] = val
                    if not first:
                        self.inbox.put(("notice", texts[(key, val)]))

    # ---- the loop -------------------------------------------------------------
    # No clock ticks. What 현 says (and notices, letters) is queued and always comes first. Otherwise the being
    # takes its next moment right away while it is doing things, and when it has nothing worth doing it checks in
    # again every idle_seconds (deciding is cheap; a model is only called when it actually does something).
    def run(self) -> None:
        last = time.time()
        busy, last_act = False, 0.0
        while True:
            if busy:
                wait = max(0.0, self.cfg["min_gap_seconds"] - (time.time() - last_act))
            else:
                wait = self.cfg["idle_seconds"]
            try:
                item = self.inbox.get(timeout=wait) if wait > 0 else self.inbox.get_nowait()
            except queue.Empty:
                item = None
            now = time.time()
            dt_inner = max(1.0, (now - last) * self.cfg["time_scale"])

            if item and item[0] == "notice":
                try:
                    with self.lock:
                        self.being.mem.set_meta("presence", self.presence)
                        rec = self.being.notice(dt_inner, item[1])
                        self.being.mem.set_meta("wall", now)
                    last = now
                    print(f"[알아챔] {item[1]}")
                except Exception as e:
                    self._fail(e)
                self._refresh()
                continue

            say_id, msg = (item[1], item[2]) if item else (None, None)
            if self.paused and msg is None:
                last, busy = now, False
                continue
            letter = None
            if msg is None:
                self._check_mail()
                if self.incoming:
                    letter = self.incoming.popleft()
            if say_id is not None:
                self._mark_pending(say_id, "listening")
                self._refresh()
            try:
                with self.lock:
                    rec = self.being.tick(dt_inner, external=msg, letter=letter)
                    if letter is not None:
                        self._mark_letter_read()
                    self.being.mem.set_meta("wall", now)
                last = now
                busy, last_act = rec["action"] != "rest", time.time()
                self.last_error = rec.get("error")
                if say_id is not None:
                    self._drop_pending(say_id)
                if rec["action"] == "explore":
                    self._count_explore()
                    print(f"[찾아봄] {rec.get('topic')} → {rec['text'][:70]}")
                elif rec["action"] == "reach_out":
                    self._count_reach()
                    self.being.outbox.clear()
                    print(f"[말 걸기] {rec['msg']}")
                    # if 현 is looking at the page, it just speaks there; otherwise it knocks
                    if self.cfg["notify"] and not self.presence["watching"]:
                        notify("깨어 있는 존재", rec["msg"])
                elif rec["action"] == "attend":
                    print(f"[들음] {rec['text'][:60]} → {rec.get('reply')}")
                if rec.get("broke"):
                    print(f"[어김] 현이 '{rec['broke']}'라고 했지만 그렇게 함")
                if rec.get("learned") and rec["learned"].get("interest"):
                    print(f"[관심] {rec['learned']['interest']} (+{rec['learned']['progress']})")
                elif rec["action"] != "rest":
                    print(f"[{rec['action']}] {rec['text'][:80]}")
            except Exception as e:
                self._fail(e)
                if msg is not None:
                    self._mark_pending(say_id, "waiting")
                    self.inbox.put(item)           # don't lose what the person said
                if letter is not None:
                    self.incoming.appendleft(letter)
                busy = False
                time.sleep(min(10, self.cfg["idle_seconds"]))
            self._refresh()

    def _fail(self, e: Exception) -> None:
        self.last_error = f"{type(e).__name__}: {e}"
        print("[오류]", self.last_error)
        traceback.print_exc()

    def _mark_pending(self, say_id, state: str) -> None:
        with self.pending_lock:
            for p in self.pending:
                if p["id"] == say_id:
                    p["state"] = state

    def _drop_pending(self, say_id) -> None:
        with self.pending_lock:
            self.pending = [p for p in self.pending if p["id"] != say_id]

    def _refresh(self) -> None:
        """Rebuild what the page sees. Done by the loop, so the page never waits on a slow organ."""
        with self.lock:
            snap = self.being.snapshot()
            snap["log"] = list(self.being.log)
            snap["can_reach"] = self._can_reach()
        if hasattr(self.backend, "vram"):
            try:
                self.vram = self.backend.vram()
            except Exception:
                self.vram = None
        self.cache = snap

    def _decider_status(self) -> dict | None:
        d = getattr(self.backend, "decider", None)
        if d is None:
            return None
        info = d.info or {}
        return {"ok": d._ok, "device": info.get("device"), "ms": d.last_ms}

    def state(self, since: int) -> dict:
        snap = dict(self.cache)
        log = snap.get("log", [])
        last = log[-1]["seq"] if log else 0
        # since = the last sequence number the page has; a larger one means the being was reborn
        snap["log"] = [e for e in log if e.get("seq", 0) > since] if since <= last else log
        snap["reset"] = since > last
        snap["last_seq"] = last
        with self.pending_lock:
            pending = [dict(p) for p in self.pending]
        snap.update(paused=self.paused, error=self.last_error, presence=dict(self.presence),
                    models=dict(loop=self.cfg["loop_model"], voice=self.cfg["voice_model"],
                                embed=self.cfg["embed_model"], backend=self.cfg["backend"]),
                    pending=pending, idle_seconds=self.cfg["idle_seconds"], vram=self.vram,
                    decider=self._decider_status())
        return snap

def make_handler(life: Life):
    page = (ROOT / "web" / "index.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console for the being's own lines
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path.startswith("/api/state"):
                if "vis=visible" in self.path:
                    life.page_seen = time.time()
                since = 0
                if "since=" in self.path:
                    try:
                        since = int(self.path.split("since=")[1].split("&")[0])
                    except ValueError:
                        since = 0
                self._json(life.state(since))
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                data = {}
            if self.path == "/api/say":
                text = str(data.get("text", "")).strip()[:500]
                if not text:
                    return self._json({"error": "빈 메시지"}, 400)
                self._json({"ok": True, "pending": life.say(text)})
            elif self.path == "/api/presence":
                self._json(life.set_presence(data.get("asleep"), data.get("watching"), data.get("save")))
            elif self.path == "/api/pause":
                life.paused = bool(data.get("paused", not life.paused))
                self._json({"paused": life.paused})
            else:
                self._json({"error": "not found"}, 404)

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--backend", choices=["ollama", "mock"])
    p.add_argument("--tick-seconds", "--idle-seconds", dest="idle_seconds", type=float)
    p.add_argument("--time-scale", type=float)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()

    cfg = dict(DEFAULTS)
    if Path(a.config).exists():
        cfg.update(json.loads(Path(a.config).read_text(encoding="utf-8")))
    if a.backend:
        cfg["backend"] = a.backend
    if a.idle_seconds:
        cfg["idle_seconds"] = a.idle_seconds
    if a.time_scale:
        cfg["time_scale"] = a.time_scale
    if a.no_browser:
        cfg["open_browser"] = False

    life = Life(cfg)
    if cfg["backend"] == "ollama":
        try:
            have = life.backend.ping()
            missing = [m for m in (cfg["loop_model"], cfg["voice_model"], cfg["embed_model"])
                       if not any(h == m or h.startswith(m + ":") or h.split(":")[0] == m for h in have)]
            if missing:
                print("[경고] Ollama에서 찾지 못한 모델:", ", ".join(missing))
                print("       설치된 모델:", ", ".join(have) or "(없음)")
                print("       config.json의 모델 이름을 'ollama list'에 나오는 이름으로 맞춰줘.")
        except Exception as e:
            print(f"[경고] Ollama에 연결하지 못했어 ({e}). Ollama가 켜져 있는지 확인해줘.")

    if cfg.get("label"):
        print(f"[{cfg['label']}] 데이터: {cfg['db']}, 편지함: {cfg['letters_dir']}, 알림: {'켬' if cfg['notify'] else '끔'}")
    print("찾아볼 수 있는 곳:", ", ".join(life.senses) or "(없음)")
    life.wake_up()
    threading.Thread(target=life.run, daemon=True).start()
    threading.Thread(target=life.sense_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", cfg["port"]), make_handler(life))
    url = f"http://localhost:{cfg['port']}"
    print(f"깨어 있음. {url} 에서 대화할 수 있어. 끄려면 Ctrl+C.")
    if cfg["open_browser"]:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n잠듦. 다음에 켜면 그동안의 공백을 느끼며 깨어날 거야.")


if __name__ == "__main__":
    main()
