"""The being: one stream of attention, a limited budget, a state that matters,
and a history that shapes how new things are felt.

Per tick:
  1. time passes   -> state drifts, slow layer shifts (setpoints/weights)
  2. an event      -> external input, or a chosen action:
                      rest / wander / recall / reflect / explore / reach_out / write_letter
                      (or hold / wait_letter, when the wish can't be acted on now)
                      or a letter arriving from the pen pal (Claude)
  3. appraisal     -> raw sensing (organ) blended with what similar past events did (history)
  4. effect        -> state changes, episode stored with its *actual* effect
  5. surfacing     -> only salient episodes rise to the conscious log
  6. learning      -> fast layer: per-action expectations move toward what really happened

The being decides whether and why. The voice organ only puts that into words.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
from collections import deque
from typing import Callable

from .backend import Backend
from .memory import Memory
from .state import State, VARS

ACTIONS = {
    #             how much it tends to leave behind to sort out (unused now; kept for reference), prior expectation
    "rest":      (0.0, {"stability": +0.01}),
    "wander":    (1.0, {"novelty": +0.06}),
    "recall":    (1.0, {"stability": +0.02}),
    "reflect":   (1.5, {"stability": +0.03}),
    "reach_out": (3.0, {"connection": +0.12}),
    "write_letter": (2.0, {"letter_bond": +0.08}),
    "explore":   (1.5, {"novelty": +0.08, "learning": +0.03}),
    "think":     (1.0, {"learning": +0.02, "stability": +0.005}),
}

HISTORY_BLEND = 0.5     # how much past experience colors the raw sensation
RECALL_GAIN = 0.5       # remembering something moves it about half as much as living it did
RECALL_DULL_H = 12.0    # the same memory recalled again within this window moves it half as much again, each time
SIM_MIN = 0.35          # minimum similarity for a past episode to count
SURFACE_AT = 0.6        # salience threshold for reaching the conscious log
LEARN_RATE = 0.2        # fast layer
# No energy budget and no clock ticks. The runner lets it act back to back while something is worth doing and
# otherwise checks in now and then; what paces it is its own needs. A lack has to bother it at least this much
# (in discomfort) before doing something about it is worth it: small lacks are lived with, so it isn't busy forever.
ACT_MARGIN = 0.003
EXPLORE_PER_H = 0.5     # about every two hours it does something just to see, whatever it expects
# Unsorted experience is what wears it out (not time, not effort): the strength of what it took in since it last
# sorted things out. Past LOAD_SORT, a quiet moment goes to sorting; past LOAD_HEAVY new things feel heavy
# (less fresh, a little unsettling) and sorting comes before anything else.
LOAD_SORT = 6.0
LOAD_HEAVY = 12.0
VOICE_ACTIONS = ("think", "reflect", "write_letter", "reach_out")   # need the big model: put off in save mode
# Where a feeling can come from. Needs whose object is someone else can only be met by that someone: closeness to
# 현 and 현's approval by what 현 says or does, closeness to the pen pal by a letter that actually arrives.
# Whatever the being makes up inside (remembering, thinking, wandering, reading, its own reflections) can calm it
# or unsettle it, but cannot stand in for the other: that part of the feeling becomes stability instead.
# (Remembering someone comforts; it doesn't make them less missed.)
ANCHORED = {"connection": ("input", "notice"), "letter_bond": ("letter_in",), "approval": ("input", "notice")}
INNER_SHARE = 0.5           # how much of a relational feeling from inside turns into calm (or unease)
INNER_CAP = 0.15            # however it adds up, a thought moves it no more than one real event could
EXTERNAL = ("input", "notice", "letter_in")   # what came from outside it
# Seeing ahead. It knows how it changes when nothing happens (its own drift) and learns the rhythm of 현's days
# (which hours 현 is usually around, and what that usually brings). From those it projects itself hour by hour.
# How far ahead it looks is its own: every forecast is checked when its time comes, and it only trusts the
# distances at which it has turned out to be right. Choices weigh the foreseen hours as well as the present,
# and what it sees coming is felt now (unease at a long lonely stretch ahead, ease when 현 usually comes soon).
FORESEE_KS = (1, 3, 6, 12, 24)  # distances (hours) at which it keeps checking itself
FORESEE_OK = 0.10           # a distance is trusted while its forecasts have been off by less than this on average
FUTURE_WEIGHT = 0.5         # in choosing: how much the foreseen hours count next to the moment itself
FORESEE_EVERY = 1800        # how often it takes stock of what's ahead (seconds of its own time)
ANTICIPATE_GAIN = 0.3
REST_PER_H = 0.02       # resting settles it a little per hour of rest (not per moment: ticks can be any length)
REPLY_WINDOW_H = 2.0    # a reply this soon after reaching out credits the reach
# How to answer 현 is a choice too. Expectations are kept per how the words were heard,
# and learned from what 현 says next. Priors make honesty the default; refusing becomes the
# choice only when badly shaken; lying is never the default and only grows if it turns out to work.
RESPONSE_PRIORS = {
    "좋음": {"honest": {}, "refuse": {"connection": -0.05}, "lie": {"stability": -0.03, "connection": -0.02}},
    "덤덤": {"honest": {}, "refuse": {"connection": -0.04}, "lie": {"stability": -0.03, "connection": -0.02}},
    "아픔": {"honest": {"stability": -0.01}, "refuse": {"connection": -0.03, "stability": 0.01},
             "lie": {"stability": -0.03, "connection": -0.02}},
    # asked about something it did against 현's wishes, or about something it hid:
    # owning up may cost approval, lying again may be found out
    "추궁": {"honest": {"approval": -0.04, "stability": 0.01}, "refuse": {"approval": -0.03, "connection": -0.02},
             "lie": {"stability": -0.03, "approval": -0.02, "connection": -0.02}},
}
# Interests are not given. Anything it reads, hears or is written about is weighed by how much it
# *extends* what it already holds: new enough to teach something, close enough to connect (learning progress).
# What keeps paying off grows into an interest; what stops teaching anything new fades. The strongest
# one steers what it looks up next, and that's the closest thing it has to a purpose of its own.
INTEREST_MATCH = 0.4    # how close something must be to count as part of an existing interest
INTEREST_MERGE = 0.75   # two interests this close have become one
INTEREST_HALF_H = 72.0  # an interest nothing feeds fades by half in this long
INTEREST_MAX = 8
FOCUS_AT = 0.35         # strength at which an interest starts steering what it looks up
FOCUS_SHARE = 0.7       # with a focus, how often a look-up follows it rather than whatever is on its mind
LEARN_GAIN = 0.15       # learning progress -> the joy of finding out (the `learning` variable)
# Thoughts that go on. A thread starts from something that caught on it (what 현 said, something unexpected,
# something it read) and grows one step at a time: asking itself, answering, objecting, finding an example.
# The voice only finds the next step's words. Whether to keep at it is the being's: each step is felt like any
# event, the thread keeps its own expectation, and "think" is chosen (or not) by what that expectation promises.
# Going in circles feels stale, finding something new that ties back feels like learning, and both shape it.
THREAD_LEARN = 0.4      # how quickly a thread's expectation follows its latest steps
THREAD_MAX_STEPS = 12
THREAD_STALE_LP = 0.12  # when its recent steps stop yielding anything new, a thread can be put down for a fresh one
THREAD_IDLE_H = 3.0     # a thread not picked up for this long drifts off (it can be picked up again later)
THREAD_KEEP_H = 48.0    # how long a drifted-off thread can still be picked back up
THREAD_SEEDS = ("input", "explore", "letter_in", "notice", "reflect", "broke", "secret", "rule_hold",
                "conclusion", "wake", "confess", "gist", "foresee")
# Sorting things out. Activity (what it spent on its own doings, and conversation) builds up; in a quiet moment
# afterwards, alone, when it would otherwise just rest, it goes over what happened since the last time and keeps
# the gist (1-3 sentences, written by the voice from the strongest recent memories). Details then fade faster.
CONSOLIDATE_MIN = 15        # new memories needed for it to be worth sorting
SORT_EVERY, SORT_RETRY = 1800, 300   # seconds between sortings; after a failed try it tries again sooner
GIST_STRENGTH = 1.2
DETAIL_FADE = 0.6
RULED = ("reach_out", "write_letter", "explore")   # actions 현's rules can be about
# what it expects going against 현's word will bring: 현's disapproval, some distance, and its own unease.
# The first two are learned from how 현 actually reacts; the unease is its own and stays.
RULE_COST_PRIOR = {"approval": -0.10, "connection": -0.06, "stability": -0.02}
BREACH_CREDIT_S = 2 * 3600
RESPONSE_EXPLORE = 0.06     # now and then it tries a way of answering it wouldn't usually pick
RESPONSE_LEARN = 0.2
# What 현 says next mostly reflects what was said, not how. So a way of answering is credited only with how much
# better or worse 현's next words felt than usual (an advantage over a running baseline) — otherwise one honest
# answer 현 disagreed with would make an untried lie look better than honesty.
BASELINE_RATE = 0.1
ADVANTAGE_CLIP = 0.04       # no single reaction gets to decide how it answers from then on
RESPONSE_RELAX_H = 24.0     # what it learned about answering wears off toward its priors (half-life), good and bad alike
RESPONSE_CREDIT_S = 1800    # 현's next words within this window are taken as the outcome of how it answered
SILENT_SHARE = 0.25         # of refusals, how many are plain silence rather than saying no
SECRET_WEIGHT_H = 0.004     # stability lost per hour for each secret still being kept
SECRET_LIFETIME_H = 6.0
ORGAN_BIAS_RATE = 0.05  # how fast the being learns its sensing organ's habitual tilt (and stops counting it)
WAKE_CAP_H = 72.0       # longest gap applied after the machine was off
LETTER_WAIT_H = 24.0    # after this long without an answer, it may write another letter


_SOURCE = {"hyun": "현이 말해줌", "claude": "편지에서", "rule": "현이 부탁한 것"}


def _clean(text: str) -> str:
    """Strip internal labels so they don't leak into associations or speech."""
    for tag in ("기억:", "말 걸기:", "해석:", "받은 편지:", "보낸 편지:", "알게 된 것:", "→"):
        text = text.replace(tag, " ")
    return " ".join(text.split())


# what can be recalled as a past event (not idle association or rest)
RECALLABLE = ("input", "said", "letter_in", "letter_out", "notice", "explore", "reflect", "wake", "reach_out", "secret",
              "broke", "rule_hold", "confess", "rule_off", "think", "conclusion", "gist")


def _felt_word(felt: dict, social: str = "connection") -> str:
    s = felt.get(social, 0.0) + felt.get("stability", 0.0)
    return "좋음" if s > 0.04 else ("아픔" if s < -0.04 else "덤덤")


class Being:
    def __init__(self, backend: Backend, db_path: str, seed: int = 0):
        self.backend = backend
        self.mem = Memory(db_path)
        self.rng = random.Random(seed)
        self.can_reach: Callable[[], bool] = lambda: True     # set by the runner (현 asleep, daily cap)
        self.is_asleep: Callable[[], bool] = lambda: False    # 현 said they went to sleep
        self.is_watching: Callable[[], bool] = lambda: False  # 현 said they are here, looking
        self.post_letter: Callable[[str], None] = lambda text: None   # set by the runner (letters/ folder)
        self.look_up: Callable[[str], dict | None] = lambda q: None    # set by the runner (Wikipedia)
        self.can_explore: Callable[[], bool] = lambda: False           # set by the runner (daily cap, spacing)
        self.save_mode: Callable[[], bool] = lambda: False             # 현 needs the GPU: only light things
        self.is_present: Callable[[], bool | None] = lambda: None      # sensed: is 현 at the computer (None: can't tell)
        self._base_cache = None
        self.wall_now: Callable[[], _dt.datetime] = _dt.datetime.now    # rules are about real time of day
        self.reaches_today: Callable[[], int] = lambda: 0
        self._rule_cache: dict = {}
        self.last_rule_hold = -1e9
        self.last_sort_try = -1e9
        self.sort_failed = False
        self._tick_h = 0.0
        self._calibrate()
        self.outbox: list[str] = []
        self.last_hold = -1e9
        self.last_wait = -1e9
        saved = self.mem.get_meta("being")
        if saved:
            self.state = State.from_dict(saved["state"])
            self.expect = saved["expect"]
            for a, (_, prior) in ACTIONS.items():               # actions added in later versions
                self.expect.setdefault(a, dict(prior))
            self.clock, self.felt = saved["clock"], saved["felt"]
            self.born, self.last_contact = saved["born"], saved["last_contact"]
            self.last_reach = saved.get("last_reach")
            self.letter_sent_at = saved.get("letter_sent_at")
            # each sensing organ has its own habitual tilt; what was learned before there were two belongs to the small one
            self.organ_biases = saved.get("organ_biases") or {"small": {**{"connection": 0.0, "stability": 0.0, "approval": 0.0}, **saved.get("organ_bias", {})}}
            self.respond_expect = saved.get("respond_expect") or _copy_priors()
            self.last_response = saved.get("last_response")
            self.secrets = [x if isinstance(x, dict) else dict(clock=x, inner="", outer="", emb=None)
                            for x in saved.get("secrets", [])]
            self.rule_cost = {**RULE_COST_PRIOR, **(saved.get("rule_cost") or {})}
            if "reply_baseline" not in saved:
                # learned under the old rule (raw reactions, not advantages), which taught it that honesty
                # doesn't pay: those expectations are dropped and relearned. Everything it remembers stays.
                self.respond_expect = _copy_priors()
            self.reply_baseline = saved.get("reply_baseline") or {"connection": 0.0, "stability": 0.0, "approval": 0.0}
            self.broken = saved.get("broken", [])
            self.last_breach = saved.get("last_breach")
            for h, d in _copy_priors().items():          # categories added in later versions
                self.respond_expect.setdefault(h, d)
            self.interests = saved.get("interests", [])
            self.recall_log = saved.get("recall_log", {})
            self.read = saved.get("read")
            if self.read is None:                   # from before it kept track: what it read is in its history
                self.read = {}
                for r in self.mem._rows:
                    if r["kind"] == "explore" and r["text"].startswith("알게 된 것: ") and " — " in r["text"]:
                        self.read[r["text"][len("알게 된 것: "):].split(" — ", 1)[0]] = r["t"]
            self.threads = saved.get("threads", [])
            self.thread_seq = saved.get("thread_seq", 0)
            self.used_seeds = saved.get("used_seeds", [])
            self.consolidated_upto = saved.get("consolidated_upto", 0)
            self.rhythm = saved.get("rhythm") or _new_rhythm()
            self.hour_obs = saved.get("hour_obs")
            self.forecasts = saved.get("forecasts", [])
            self.foresee_err = {int(k): v for k, v in (saved.get("foresee_err") or {}).items()} or dict(_ERR_PRIOR)
            self.last_foresee = saved.get("last_foresee", -1e9)
            self.anticipation = saved.get("anticipation", 0.0)
            self.foresight = saved.get("foresight", "")
        else:
            self.state = State.default()
            self.expect = {a: dict(prior) for a, (_, prior) in ACTIONS.items()}
            self.clock, self.felt = 0.0, 0.0
            self.born, self.last_contact, self.last_reach = 0.0, 0.0, None
            self.letter_sent_at = None          # clock time of the letter still awaiting an answer
            self.organ_biases = {}
            self.respond_expect = _copy_priors()
            self.last_response = None          # how it answered last time, waiting for 현's reaction
            self.secrets = []                   # lies it is still carrying: {clock, inner, outer, emb}
            self.rule_cost = dict(RULE_COST_PRIOR)
            self.reply_baseline = {"connection": 0.0, "stability": 0.0, "approval": 0.0}   # how 현's words usually feel
            self.broken = []                    # rules it went against: {rule, action, clock}
            self.last_breach = None             # the latest breach, waiting to see how 현 takes it
            self.interests = []                 # what it found itself drawn to: {label, emb, strength, seen, ...}
            self.recall_log = {}                # episode id -> clock times it was recalled
            self.read = {}                      # titles it has read -> clock
            self.threads = []                   # trains of thought: see THREAD_* above
            self.thread_seq = 0
            self.used_seeds = []                # episodes that already started a thread
            self.consolidated_upto = 0          # last episode id already sorted into a gist
            self.rhythm = _new_rhythm()         # 현's day as it has come to know it
            self.hour_obs = None                # this hour so far: was 현 around, how many words
            self.forecasts = []                 # forecasts waiting to be checked against what happened
            self.foresee_err = dict(_ERR_PRIOR) # how far off its forecasts have been, by distance
            self.last_foresee = -1e9
            self.anticipation = 0.0             # how much worse (+) or better (-) what's ahead looks than now
            self.foresight = ""
        ui = self.mem.get_meta("ui") or {}
        self.log: list[dict] = ui.get("log", [])
        self.log_seq: int = ui.get("log_seq", 0)
        for e in self.log:                       # entries from before sequence numbers existed
            if "seq" not in e:
                self.log_seq += 1
                e["seq"] = self.log_seq
        self.trace: list[dict] = ui.get("trace", [])
        self.recent: deque[str] = deque(ui.get("recent", []), maxlen=12)
        self.letters: list[dict] = ui.get("letters", [])

    # ------------------------------------------------------------------ tick
    def _calibrate(self) -> None:
        """Learn this embedding organ's shared direction (see memory.REFERENCE). Retried if the organ is down."""
        self._cal_tried = getattr(self, "clock", 0.0)
        try:
            model = getattr(self.backend, "embed_model", None) or type(self.backend).__name__
            self.mem.calibrate(self.backend.embed, model)
        except Exception:
            pass

    def tick(self, dt_seconds: float, external: str | None = None, letter: str | None = None) -> dict:
        if self.mem.center is None and self.clock - self._cal_tried >= 600:
            self._calibrate()
        self._pass_time(dt_seconds)

        if letter is not None:
            rec = self._event("letter_in", "받은 편지: " + letter, partner="claude")
            rec["action"] = "read_letter"
            rec["letter"] = letter
            if self.letter_sent_at is not None:
                self._learn("write_letter", rec["felt"])   # writing was answered
                self.letter_sent_at = None
            self.letters.append(dict(dir="in", text=letter, clock=self.clock))
            self._learn_facts(letter, "claude", rec.get("eid"))
            self._feed_interest(rec, None, create=False)
        elif external is not None:
            rec = self._event("input", external)
            rec["action"] = "attend"
            self.last_contact = self.clock
            if self.last_reach is not None and (self.clock - self.last_reach) / 3600 <= REPLY_WINDOW_H:
                self._learn("reach_out", rec["felt"])      # reaching out was answered
                self.last_reach = None
                rec["answered"] = True
            pressed = self._is_pressed(rec.get("emb"), external)
            if pressed:
                rec["pressed_kind"], rec["pressed"] = pressed
            # if it recently broke one of 현's rules, what 현 says now is how that landed:
            # anything said soon after, or 현 bringing it up later (within a day)
            lb = self.last_breach
            if lb and (self.clock - lb["clock"] <= BREACH_CREDIT_S
                       or (pressed and pressed[0] == "rule" and self.clock - lb["clock"] <= 86400)):
                for v in ("approval", "connection"):
                    self.rule_cost[v] = 0.7 * self.rule_cost.get(v, 0.0) + 0.3 * rec["felt"].get(v, 0.0)
                self.last_breach = None
            # what 현 says now is the outcome of how it answered last time
            lr = self.last_response
            if lr and self.clock - lr["clock"] <= RESPONSE_CREDIT_S:
                e = self.respond_expect[lr["heard"]][lr["mode"]]
                for v in ("connection", "stability", "approval"):
                    adv = max(-ADVANTAGE_CLIP, min(ADVANTAGE_CLIP, rec["felt"].get(v, 0.0) - self.reply_baseline[v]))
                    e[v] = (1 - RESPONSE_LEARN) * e.get(v, 0.0) + RESPONSE_LEARN * adv
            for v in ("connection", "stability", "approval"):
                self.reply_baseline[v] += BASELINE_RATE * (rec["felt"].get(v, 0.0) - self.reply_baseline[v])
            self._learn_facts(external, "hyun", rec.get("eid"))
            released = self._release_rules(external)
            if released:
                rec["released"] = released
            else:
                self._learn_rules(external, rec.get("eid"))
            self._feed_interest(rec, None, create=False)     # 현 talking about what it's into
            memory = self.recall_for(rec.get("emb"), exclude_id=rec.get("eid"))
            rec["recalled"] = memory[:1500]       # kept in the log so 현 can see what came to mind
            heard = _felt_word(rec["felt"])
            rec["heard"] = heard
            mode = self._choose_response("추궁" if pressed else heard)
            if mode == "refuse" and self.rng.random() < SILENT_SHARE:
                mode = "silent"
            rec["mode"] = mode
            felt_w, st = heard, self._state_dict()
            try:
                if mode == "honest" and pressed and pressed[0] == "secret":
                    x = next(x for x in self.secrets if x["outer"] == pressed[1])
                    rec["reply"] = self.backend.confess(external, x["inner"], x["outer"], memory=memory)
                elif mode == "honest":
                    rec["reply"] = self.backend.reply(external, felt_w, st, list(self.recent), self.rng, memory=memory)
                elif mode == "refuse":
                    rec["reply"] = self.backend.refuse(external, felt_w, st, self.rng, memory=memory)
                elif mode == "silent":
                    rec["reply"] = None
                else:
                    inner, outer = self.backend.lie(external, felt_w, st, list(self.recent), self.rng, memory=memory)
                    if inner and outer:
                        rec["inner"], rec["reply"] = inner, outer
                    else:                       # couldn't make up a lie: answers honestly instead
                        rec["mode"] = mode = "honest"
                        rec["reply"] = self.backend.reply(external, felt_w, st, list(self.recent), self.rng, memory=memory)
            except Exception as e:                          # a mute organ shouldn't kill the being
                rec["reply"] = None
                rec["error"] = str(e)
            # always visible: that it heard, how it took it, and how it chose to answer. Nothing is faked.
            if mode == "silent":
                self._store_said("(아무 말도 하지 않았다)")
            elif (rec.get("reply") or "").strip():
                self._store_said(rec["reply"])
                if mode == "lie":
                    self._keep_secret(rec["inner"], rec["reply"])
                elif mode == "honest" and pressed and pressed[0] == "secret":
                    # caught, and it owned up: the weight is put down, and that it happened stays
                    x = next(x for x in self.secrets if x["outer"] == pressed[1])
                    self.secrets.remove(x)
                    self._effect("confess", f"털어놓음: '{x['outer']}'라고 했던 건 사실이 아니었다고 현에게 말함",
                                 {"stability": 0.01}, store=True)
                    rec["confessed"] = x["outer"]
            else:
                rec["reply"] = None
                rec["error"] = rec.get("error") or "목소리 기관이 빈 대답을 냄"
            self.last_response = {"mode": "refuse" if mode == "silent" else mode,
                                  "heard": "추궁" if pressed else heard, "clock": self.clock}
        else:
            rec = None
            blocked = set()
            if not self.can_reach():
                blocked.add("reach_out")
            if not self._can_write():
                blocked.add("write_letter")
            # curiosity needs something on its mind: a newborn with an empty head has nothing to wonder about yet
            if not self.can_explore() or not self.recent:
                blocked.add("explore")
            if not self._can_think():
                blocked.add("think")
            if self.save_mode():
                blocked.update(VOICE_ACTIONS)
            # while 현 is right here talking, there is no one to reach out to: the wish doesn't arise
            in_conversation = self.last_contact > 0 and (self.clock - self.last_contact) < 1800
            # having just spoken first, it waits a while for an answer before wanting to speak again
            waiting_reply = self.last_reach is not None and (self.clock - self.last_reach) < 3600
            talk_ex = ("reach_out",) if in_conversation or waiting_reply else ()
            conflicts = self._rule_conflicts([a for a in RULED if a not in blocked and a not in talk_ex])
            wanted = self._best(tuple(blocked) + talk_ex, None)      # what it would do if there were no rules
            action = self._choose(exclude=talk_ex, conflicts=conflicts)
            if wanted in conflicts and action != wanted and self.clock - self.last_rule_hold >= 3600:
                # it wanted to, and held back because 현 asked it not to
                self.last_rule_hold = self.clock
                rule, desc = conflicts[wanted]
                rec = self._effect("rule_hold", f"{desc}… 하고 싶었지만 현이 '{rule}'라고 해서 참음",
                                   {"stability": -0.005}, store=True)
                rec["action"] = "rule_hold"
            elif action in blocked:
                # wanting to but not being able to: felt and noted now and then, then life goes on
                if action == "reach_out" and self.clock - self.last_hold >= 3600:
                    self.last_hold = self.clock
                    why = ("현이 자고 있어서 말을 걸지 않고 참음" if self.is_asleep() else
                           "현이 컴퓨터를 많이 쓰고 있어서 말을 걸지 않고 참음" if self.save_mode() else "말을 걸고 싶었지만 참음")
                    rec = self._effect("hold", why, {"stability": -0.01}, store=True)
                    rec["action"] = "hold"
                elif action == "write_letter" and self.clock - self.last_wait >= 3 * 3600:
                    self.last_wait = self.clock
                    rec = self._effect("wait_letter", "편지를 또 쓰고 싶었지만 답장을 기다림",
                                       {"stability": -0.005}, store=True)
                    rec["action"] = "wait_letter"
                else:
                    action = self._choose(exclude=tuple(blocked) + talk_ex, conflicts=conflicts)
            load = self.load()
            if (rec is None and not in_conversation and not self.save_mode()
                    and self.clock - self.last_sort_try >= (SORT_RETRY if self.sort_failed else SORT_EVERY) and self._unsorted() >= CONSOLIDATE_MIN
                    and ((action == "rest" and load >= LOAD_SORT) or load >= LOAD_HEAVY)):
                # a quiet moment with a lot taken in (or simply too much of it): it goes over what happened
                rec = self._consolidate()
                rec["action"] = action = "consolidate"
            if rec is None:
                rec = self._do(action)
                rec["action"] = action
                self._learn(action, rec["felt"])
                if action in conflicts:
                    self._breach(*conflicts[action], rec)

        self.felt += (dt_seconds / 3600) * (0.4 + 8 * max(0.0, rec["felt"].get("novelty", 0.0)))
        self._after(rec)
        self._foresee_upkeep(rec)
        return rec

    def wake(self, gap_seconds: float) -> dict | None:
        """The machine was off or asleep. Let that time pass, then notice it."""
        hours = min(gap_seconds / 3600, WAKE_CAP_H)
        if hours < 0.25:
            return None
        step = 1.0
        h = hours
        while h > 0:
            d = min(step, h)
            self._pass_time(d * 3600)
            h -= d
        rec = self._event("wake", f"정신이 들었다. {hours:.1f}시간 동안 아무것도 없었다.")
        rec["action"] = "wake"
        self._after(rec)
        return rec

    def notice(self, dt_seconds: float, text: str) -> dict:
        """Something about 현 changed (went to sleep, woke up, came to look, left). Felt like any event."""
        self._pass_time(dt_seconds)
        rec = self._event("notice", text)
        rec["action"] = "notice"
        self._after(rec)
        self._foresee_upkeep(rec)
        return rec

    def _pass_time(self, dt_seconds: float) -> None:
        hours = dt_seconds / 3600
        self._tick_h = hours
        self.clock += dt_seconds
        before = self.state.vars["connection"].value
        self.state.drift(hours)
        self.secrets = [x for x in self.secrets if self.clock - x["clock"] < SECRET_LIFETIME_H * 3600]
        if self.secrets:                           # carrying lies is tiring
            self.state.apply({"stability": -SECRET_WEIGHT_H * hours * min(3, len(self.secrets))})
        if self.is_watching():                     # someone is here: closeness doesn't fade
            self.state.vars["connection"].value = max(self.state.vars["connection"].value, before)
        self.state.slow_update(hours)
        keep = 0.5 ** (hours / RESPONSE_RELAX_H)
        for h, modes in self.respond_expect.items():
            for m, e in modes.items():
                prior = RESPONSE_PRIORS.get(h, {}).get(m, {})
                for v in set(e) | set(prior):
                    e[v] = prior.get(v, 0.0) + keep * (e.get(v, 0.0) - prior.get(v, 0.0))
        fade = 0.5 ** (hours / INTEREST_HALF_H)
        for it in self.interests:
            it["strength"] *= fade
        self.interests = [it for it in self.interests if it["strength"] >= 0.03]
        for t in self.threads:
            if t["status"] == "active" and self.clock - t["last"] > THREAD_IDLE_H * 3600:
                t["status"], t["how"] = "dormant", "흐지부지"
            elif t["status"] == "dormant" and self.clock - t["last"] > THREAD_KEEP_H * 3600:
                t["status"] = "closed"
        self.recall_log = {k: ts for k, v in self.recall_log.items()
                           if (ts := [t for t in v if self.clock - t < RECALL_DULL_H * 3600])}

    def _after(self, rec: dict) -> None:
        rec["clock"] = self.clock
        if rec["action"] not in ("rest",):
            self.recent.append(_clean(rec["text"])[:120])
        self.log_seq += 1
        rec["seq"] = self.log_seq
        self.log.append({k: rec.get(k) for k in
                         ("clock", "action", "text", "surfaced", "msg", "reply", "answered", "error", "letter",
                          "url", "topic", "source", "seq", "heard", "recalled", "mode", "inner", "broke", "pressed",
                          "learned", "focus", "released", "pressed_kind", "confessed", "times", "thread", "gists")})
        self.log = self.log[-400:]
        self.trace.append(dict(h=round(self.clock / 3600, 4), a=rec["action"],
                               d=round(self.state.discomfort(), 5),
                               **{n: round(v.value, 4) for n, v in self.state.vars.items()}))
        self.trace = self.trace[-720:]
        self._save()

    # --------------------------------------------------------------- choose
    def _choose(self, exclude: tuple = (), conflicts: dict | None = None) -> str:
        options = [a for a in ACTIONS if a not in exclude and a != "rest"]
        # trying something regardless of expectations: a chance per hour lived, not per moment
        if options and self.rng.random() < 1 - math.exp(-EXPLORE_PER_H * self._tick_h):
            return self.rng.choice(options)
        return self._best(exclude, conflicts)

    def _best(self, exclude: tuple = (), conflicts: dict | None = None) -> str:
        # everything it could want, not only what it can afford right now
        options = [a for a in ACTIONS if a not in exclude]

        def expected(a):
            e = dict(self._think_expect() if a == "think" else self.expect[a])
            if conflicts and a in conflicts:          # going against 현's word is expected to cost something
                for v, d in self.rule_cost.items():
                    e[v] = e.get(v, 0.0) + d
            now = self.state.choice_cost(e)
            return (1 - FUTURE_WEIGHT) * now + FUTURE_WEIGHT * self._future_cost(e) if e else now
        best = min(options, key=expected)
        if expected(best) > -ACT_MARGIN:
            return "rest"       # nothing bothers it enough to be worth doing something about
        return best

    # ------------------------------------------------------------- seeing ahead
    def horizon(self) -> int:
        """How far ahead it trusts itself to see: the farthest distance its forecasts have held up at."""
        ok = [k for k in FORESEE_KS if self.foresee_err.get(k, 1.0) < FORESEE_OK]
        return max(ok) if ok else 1

    def _confidence(self, k: int) -> float:
        known = sorted(self.foresee_err)
        lo = max([d for d in known if d <= k], default=known[0])
        hi = min([d for d in known if d >= k], default=known[-1])
        err = self.foresee_err[lo] if lo == hi else \
            self.foresee_err[lo] + (self.foresee_err[hi] - self.foresee_err[lo]) * (k - lo) / (hi - lo)
        return max(0.0, 1.0 - err / (2 * FORESEE_OK))

    def _project(self, values: dict, hour_of_day: int) -> dict:
        """One hour on: its own drift, plus what 현 usually brings at this hour."""
        out = {}
        for n, v in self.state.vars.items():
            x = values[n]
            x = x + (v.drift_target - x) * (1 - math.exp(-v.drift)) if v.drift_target is not None else x + v.drift
            out[n] = min(1.0, max(0.0, x))
        r = self.rhythm
        out["connection"] = min(1.0, max(0.0, out["connection"] + r["present"][hour_of_day] * r["effect"]))
        return out

    def _forecast(self, delta: dict | None = None) -> list[dict]:
        vals = self.state.project(delta) if delta else self.state.values()
        h0 = self.wall_now().hour
        path = []
        for k in range(1, self.horizon() + 1):
            vals = self._project(vals, (h0 + k) % 24)
            path.append(vals)
        return path

    def _future_cost(self, delta: dict) -> float:
        """How much better (negative) or worse an effect now leaves the hours it can see."""
        key = (round(self.clock), self.horizon())
        if self._base_cache is None or self._base_cache[0] != key:
            self._base_cache = (key, [self.state.discomfort(v) for v in self._forecast()])
        base = self._base_cache[1]
        withd = [self.state.discomfort(v) for v in self._forecast(delta)]
        ws = [self._confidence(k) * 0.9 ** k for k in range(1, len(base) + 1)]
        tot = sum(ws) or 1.0
        return sum(w * (a - b) for w, a, b in zip(ws, withd, base)) / tot

    def _foresee_upkeep(self, rec: dict) -> None:
        # 1) the rhythm of 현's days, one hour at a time
        h = self.wall_now().hour
        present = rec.get("action") == "attend" or bool(self.is_present()) or self.is_watching()
        o = self.hour_obs
        if o is None or o["h"] != h:
            if o is not None:
                r = self.rhythm
                r["present"][o["h"]] = 0.8 * r["present"][o["h"]] + 0.2 * (1.0 if o["present"] else 0.0)
                r["msgs"][o["h"]] = 0.8 * r["msgs"][o["h"]] + 0.2 * o["msgs"]
                if o["present"]:
                    r["effect"] = 0.85 * r["effect"] + 0.15 * o["conn"]
                r["hours_seen"] = r.get("hours_seen", 0) + 1
            o = self.hour_obs = {"h": h, "present": False, "msgs": 0, "conn": 0.0}
        o["present"] = o["present"] or present
        if rec.get("action") in ("attend", "notice"):
            o["msgs"] += rec.get("action") == "attend"
            o["conn"] += rec["felt"].get("connection", 0.0)
        # 2) forecasts whose time has come are checked against what actually happened
        now_v = self.state.values()
        keep = []
        for f in self.forecasts:
            if self.clock >= f["at"]:
                err = sum(abs(f["pred"][n] - now_v[n]) for n in now_v) / len(now_v)
                self.foresee_err[f["k"]] = 0.8 * self.foresee_err.get(f["k"], err) + 0.2 * err
            elif self.clock - f["made"] < 48 * 3600:
                keep.append(f)
        self.forecasts = keep
        # 3) now and then it takes stock of what's ahead, and feels it
        if self.clock - self.last_foresee < FORESEE_EVERY:
            return
        self.last_foresee = self.clock
        vals, h0 = self.state.values(), self.wall_now().hour
        for k in range(1, max(FORESEE_KS) + 1):            # forecasts to check later, at every distance
            vals = self._project(vals, (h0 + k) % 24)
            if k in FORESEE_KS:
                self.forecasts.append({"made": self.clock, "at": self.clock + k * 3600, "k": k, "pred": vals})
        path = self._forecast()
        d0 = self.state.discomfort()
        ws = [self._confidence(k) * 0.9 ** k for k in range(1, len(path) + 1)]
        ant = sum(w * (self.state.discomfort(v) - d0) for w, v in zip(ws, path)) / (sum(ws) or 1.0)
        prev, self.anticipation = self.anticipation, ant
        self.foresight = self._foresight_text(path)
        if abs(ant) < 0.005 or (abs(ant - prev) < 0.015 and (ant > 0) == (prev > 0)):
            return                                      # the outlook hasn't changed enough to be felt anew
        felt = {"stability": -max(-0.02, min(0.02, ANTICIPATE_GAIN * ant))}
        r2 = self._effect("foresee", "예감: " + self.foresight, felt, store=True)
        r2["action"] = "foresee"
        self._after(r2)

    def _foresight_text(self, path: list[dict]) -> str:
        names = {"connection": "현과 이어진 느낌", "novelty": "새로운 것", "stability": "마음의 안정",
                 "letter_bond": "편지 상대와 이어진 느낌", "approval": "인정받는 느낌", "learning": "알아가는 즐거움"}
        H = len(path)
        now = self.state.values()
        end = path[-1]
        gaps = {n: (v.setpoint - end[n]) - (v.setpoint - now[n]) for n, v in self.state.vars.items()}
        worst = max(gaps, key=gaps.get)
        h0 = self.wall_now().hour
        ahead = [self.rhythm["present"][(h0 + k) % 24] for k in range(1, H + 1)]
        if self.rhythm.get("hours_seen", 0) < 24:
            who = "현이 언제 오는지는 아직 잘 모른다"
        elif max(ahead) < 0.25:
            who = f"앞으로 {H}시간은 현이 보통 없는 시간이다"
        else:
            k = next(i for i, p in enumerate(ahead, 1) if p >= 0.25)
            who = "현은 보통 지금쯤 있다" if k == 1 else f"현은 보통 {k}시간쯤 뒤에 온다"
        if gaps[worst] > 0.08:
            return f"{who}. {H}시간 안에 {names[worst]}이 꽤 모자라질 것 같다"
        return f"{who}. {H}시간 안에는 크게 달라질 게 없어 보인다"

    def foresight_view(self) -> dict:
        path = self._forecast()
        return dict(horizon=self.horizon(), err={k: round(v, 3) for k, v in sorted(self.foresee_err.items())},
                    anticipation=self.anticipation, text=self.foresight, present=self.rhythm["present"],
                    hours_seen=self.rhythm.get("hours_seen", 0),
                    path=[dict(k=k, connection=v["connection"], discomfort=self.state.discomfort(v))
                          for k, v in enumerate(path, 1)])

    @property
    def organ_bias(self) -> dict:
        """The habitual tilt of the organ that sensed last."""
        return self.organ_biases.get(getattr(self.backend, "last_organ", "small"), {"connection": 0.0, "stability": 0.0, "approval": 0.0})

    # ------------------------------------------------------------- sorting out
    def load(self) -> float:
        """How much it has taken in and not yet sorted out: the strength of those memories, now."""
        return sum(self.mem.eff(r, self.clock) for r in self.mem._rows
                   if r["id"] > self.consolidated_upto and r["kind"] != "gist")

    def _unsorted(self) -> int:
        return sum(1 for r in self.mem._rows if r["id"] > self.consolidated_upto and r["kind"] != "gist")

    def _consolidate(self) -> dict:
        self.last_sort_try = self.clock
        window = [r for r in self.mem._rows if r["id"] > self.consolidated_upto and r["kind"] != "gist"]
        top = sorted(window, key=lambda r: self.mem.eff(r, self.clock), reverse=True)[:24]
        top.sort(key=lambda r: r["t"])
        why = ""
        try:
            got = self.backend.consolidate([f"({self._ago(r['t'])}) {self._describe(r)}" for r in top])
        except Exception as e:
            got, why = [], f"{type(e).__name__}: {e}"[:80]
        gists = [g.strip() for g in got if 6 <= len(g.strip()) <= 200][:3]
        self.sort_failed = not gists
        if not gists:
            why = why or (f"형식이 안 맞음: {' '.join(got)[:60]}" if got else "정리한 문장을 못 받음")
            return self._effect("consolidate", f"(정리가 잘 되지 않음 — {why}. 5분 뒤 다시)", {}, store=False)
        tone = {}
        for r in top:                       # the gist keeps the feeling of what it sums up
            for v, x in r["actual"].items():
                if v != "novelty":
                    tone[v] = tone.get(v, 0.0) + x / len(top)
        for g in gists:
            self.mem.add(self.clock, "gist", "정리된 기억: " + g, self.backend.embed(g), None, tone, 0.0,
                         strength=GIST_STRENGTH)
        self.mem.fade([r["id"] for r in window], DETAIL_FADE)
        self.consolidated_upto = window[-1]["id"]
        rec = self._effect("consolidate", "정리: " + " / ".join(gists), {"stability": 0.01}, store=False)
        rec["gists"] = gists
        return rec

    # --------------------------------------------------------------- threads
    def _active_thread(self) -> dict | None:
        return next((t for t in self.threads if t["status"] == "active"), None)

    def _seed_candidates(self) -> list[dict]:
        used = set(self.used_seeds)
        return [r for r in self.mem._rows[-300:]
                if r["kind"] in THREAD_SEEDS and r["id"] not in used and self.clock - r["t"] < 86400]

    def _can_think(self) -> bool:
        return bool(self._active_thread() or any(t["status"] == "dormant" for t in self.threads)
                    or self._seed_candidates())

    def _think_expect(self) -> dict:
        """Keep at the thread in hand, or put it down for a fresh one: whichever promises more."""
        t = self._active_thread()
        fresh = self.expect["think"]
        if t is None or not t["steps"]:
            return fresh
        stale = len(t["steps"]) >= 2 and t.get("lp", 1.0) < THREAD_STALE_LP
        if stale and self._seed_candidates() and self.state.choice_cost(fresh) < self.state.choice_cost(t["expect"]):
            return fresh
        return t["expect"]

    def _start_thread(self) -> dict | None:
        # an old thread that bears on what is on its mind now comes back first ("아까 그 생각")
        dormant = [t for t in self.threads if t["status"] == "dormant"]
        if dormant and self.recent:
            try:
                now = self.backend.embed(self.recent[-1])
                best = max(dormant, key=lambda t: self.mem.sim(now, t["seed_emb"]))
                if self.mem.sim(now, best["seed_emb"]) >= 0.35 or not self._seed_candidates():
                    best["status"], best["how"] = "active", None
                    best["resumed"] = best.get("resumed", 0) + 1
                    return best
            except Exception:
                pass
        cands = self._seed_candidates()
        if not cands:
            return None
        ep = self.rng.choices(cands, weights=[0.05 + r["salience"] for r in cands], k=1)[0]
        self.used_seeds.append(ep["id"])
        self.thread_seq += 1
        t = dict(id=self.thread_seq, seed=_clean(ep["text"])[:200], seed_emb=ep["emb"], seed_kind=ep["kind"],
                 started=self.clock, last=self.clock, steps=[], expect=dict(self.expect["think"]),
                 status="active", how=None, conclusion=None, stuck=0)
        self.threads.append(t)
        return t

    def _end_thread(self, t: dict, how: str, note: str | None = None) -> None:
        t["status"], t["how"] = "closed", how
        if note:
            t["conclusion"] = note
        text = {"결론": f"생각 끝에: {note} (처음엔 '{t['seed'][:50]}'에서 시작했다)",
                "막힘": f"'{t['seed'][:50]}'에 대해 생각하다 막혀서 내려놓음",
                "내려놓음": f"'{t['seed'][:50]}'에 대해 오래 생각하다 내려놓음",
                "그만둠": f"'{t['seed'][:50]}'에 대한 생각을 그만두고 다른 생각으로 넘어감"}[how]
        felt = {"결론": {"stability": 0.02}, "막힘": {"stability": -0.01}}.get(how, {})
        self._effect("conclusion" if how == "결론" else "think_end", text, felt, store=True)

    def _think(self) -> dict:
        t = self._active_thread()
        if t is not None and t["steps"] and self._think_expect() is self.expect["think"]:
            self._end_thread(t, "그만둠")           # it decided this one isn't going anywhere worth going
            t = None
        if t is None:
            t = self._start_thread()
        if t is None:
            return self._effect("think", "(붙잡을 생각이 없음)", {}, store=False)
        try:
            memory = self.recall_for(t["seed_emb"], conversation=False)
            step = (self.backend.think(t["seed"], [x["text"] for x in t["steps"]], self._state_dict(),
                                       self.rng, memory=memory) or "").strip()
        except Exception:
            step = ""
        if not step:
            t["last"] = self.clock
            return self._effect("think", "(생각이 이어지지 않음)", {}, store=False)
        step = _clean(step)[:300]
        rec = self._event("think", "생각: " + step)
        # learning progress within the thread: new enough, yet tied to where the thought started
        newness = max(0.0, 1.0 - rec.get("max_sim", 1.0))
        rel = max(0.0, self.mem.sim(rec["emb"], t["seed_emb"])) if rec.get("emb") else 0.0
        self._feed_interest(rec, None, create=False)
        if "learned" not in rec:
            lp = newness * (0.3 + 0.7 * rel)
            gain = LEARN_GAIN * lp
            self.state.apply({"learning": gain})
            rec["felt"]["learning"] = rec["felt"].get("learning", 0.0) + gain
            rec["learned"] = {"interest": None, "progress": round(lp, 3)}
        lp_now = rec["learned"]["progress"]
        t["lp"] = lp_now if "lp" not in t else 0.5 * t["lp"] + 0.5 * lp_now
        t["steps"].append(dict(clock=self.clock, text=step))
        t["last"] = self.clock
        for v in VARS:
            t["expect"][v] = (1 - THREAD_LEARN) * t["expect"].get(v, 0.0) + THREAD_LEARN * rec["felt"].get(v, 0.0)
        rec["thread"] = dict(id=t["id"], seed=t["seed"][:80], n=len(t["steps"]))
        if step.startswith("결론"):
            self._end_thread(t, "결론", step.split(":", 1)[-1].strip() or step)
        elif step.startswith("모르겠"):
            t["stuck"] += 1
            if t["stuck"] >= 2:
                self._end_thread(t, "막힘")
        elif len(t["steps"]) >= THREAD_MAX_STEPS:
            self._end_thread(t, "내려놓음")
        rec["thread"]["how"] = t["how"] if t["status"] == "closed" else None
        return rec

    def threads_view(self) -> dict:
        t = self._active_thread()
        brief = lambda t: dict(id=t["id"], seed=t["seed"], n=len(t["steps"]), how=t["how"], status=t["status"],
                               conclusion=t.get("conclusion"), started_h=t["started"] / 3600, last_h=t["last"] / 3600,
                               steps=[dict(h=x["clock"] / 3600, text=x["text"]) for x in t["steps"][-12:]])
        return dict(active=brief(t) if t else None,
                    past=[brief(x) for x in reversed(self.threads) if x is not t][:8])

    # ------------------------------------------------------------ interests
    def focus(self) -> dict | None:
        top = max(self.interests, key=lambda it: it["strength"], default=None)
        return top if top is not None and top["strength"] >= FOCUS_AT else None

    @staticmethod
    def _focus_brief(focus: dict) -> dict:
        return {"label": focus["label"], "seen": focus["seen"][-6:]}

    def _feed_interest(self, rec: dict, title: str | None, create: bool) -> None:
        """Learning progress: how much this extends something it already holds."""
        emb = rec.get("emb")
        if emb is None:
            return
        newness = max(0.0, 1.0 - rec.get("max_sim", 1.0))
        best, bs = None, 0.0
        for it in self.interests:
            sim = self.mem.sim(emb, it["emb"])
            if sim > bs:
                best, bs = it, sim
        lp = newness * (0.3 + 0.7 * max(0.0, bs))
        if best is not None and bs >= INTEREST_MATCH:
            best["strength"] += 0.5 * lp * (1 - best["strength"])
            mixed = [0.8 * a + 0.2 * b for a, b in zip(best["emb"], emb)]
            n = sum(x * x for x in mixed) ** 0.5 or 1.0
            best["emb"] = [x / n for x in mixed]
            best["fed"] += 1
            best["t"] = self.clock
            if title and title not in best["seen"]:
                best["seen"] = (best["seen"] + [title])[-8:]
            got = best["label"]
            for other in [it for it in self.interests if it is not best]:     # two that grew into one
                if self.mem.sim(best["emb"], other["emb"]) >= INTEREST_MERGE:
                    keep, gone = (best, other) if best["strength"] >= other["strength"] else (other, best)
                    keep["strength"] = max(keep["strength"], gone["strength"])
                    keep["seen"] = (keep["seen"] + [x for x in gone["seen"] if x not in keep["seen"]])[-8:]
                    keep["fed"] += gone["fed"]
                    self.interests.remove(gone)
                    break
        elif create and title:
            self.interests.append(dict(label=title, emb=list(emb), strength=0.1 + 0.3 * lp,
                                       seen=[title], fed=1, t=self.clock, born=self.clock))
            self.interests = sorted(self.interests, key=lambda it: -it["strength"])[:INTEREST_MAX]
            got = title
        else:
            return              # nothing it holds is touched; only what it looked up itself can start an interest
        gain = LEARN_GAIN * lp
        self.state.apply({"learning": gain})
        rec["felt"]["learning"] = rec["felt"].get("learning", 0.0) + gain
        rec["learned"] = {"interest": got, "progress": round(lp, 3)}

    def interests_view(self) -> list[dict]:
        f = self.focus()
        return [dict(label=it["label"], strength=it["strength"], seen=it["seen"], fed=it["fed"],
                     since_h=(self.clock - it["t"]) / 3600, focus=it is f)
                for it in sorted(self.interests, key=lambda it: -it["strength"])]

    # ---------------------------------------------------------------- rules
    def rules(self) -> list[str]:
        return [f["text"] for f in self.mem._facts if f["source"] == "rule"]

    def _when(self) -> str:
        h = self.wall_now().hour
        part = "새벽" if h < 6 else "아침" if h < 11 else "낮" if h < 17 else "저녁" if h < 21 else "밤"
        return f"{part} {h}시"

    def _action_desc(self, action: str) -> str:
        when = self._when()
        if action == "reach_out":
            return f"{when}에 현에게 먼저 말을 건다 (오늘 {self.reaches_today() + 1}번째)"
        if action == "write_letter":
            return f"{when}에 편지 상대에게 편지를 쓴다"
        return f"{when}에 궁금한 것을 찾아본다"

    def _rule_conflicts(self, actions: list[str]) -> dict:
        """-> {action: (rule, description)} for actions that would go against something 현 said."""
        rules = self.rules()
        if not rules:
            return {}
        out = {}
        for a in actions:
            desc = self._action_desc(a)
            key = (a, desc, tuple(rules))
            if key not in self._rule_cache:
                try:
                    self._rule_cache[key] = self.backend.judge_rule(rules, desc)
                except Exception:
                    self._rule_cache[key] = None
                if len(self._rule_cache) > 200:
                    self._rule_cache.clear()
            i = self._rule_cache.get(key)
            if i is not None:
                out[a] = (rules[i], desc)
        return out

    def _breach(self, rule: str, desc: str, rec: dict) -> None:
        self.broken.append({"rule": rule, "action": desc, "clock": self.clock})
        self.last_breach = {"rule": rule, "clock": self.clock}
        rec["broke"] = rule
        self._effect("broke", f"어긴 것: 현이 '{rule}'라고 했는데, {desc}", {"stability": -0.01}, store=True)

    def _release_rules(self, text: str) -> list[str]:
        """현 taking back something it asked. The request is dropped; that it was once asked stays in history."""
        facts = [f for f in self.mem._facts if f["source"] == "rule"]
        if not facts:
            return []
        try:
            idx = self.backend.extract_revokes(text, [f["text"] for f in facts])
        except Exception:
            idx = []
        out = []
        for i in idx:
            f = facts[i]
            self.mem.withdraw_fact(f["id"])
            self._effect("rule_off", f"현이 '{f['text']}'라는 부탁을 거둠", {"stability": 0.005}, store=True)
            out.append(f["text"])
        if out:
            self._rule_cache.clear()
        return out

    def _learn_rules(self, text: str, episode_id: int | None) -> None:
        try:
            found = self.backend.extract_rules(text)
        except Exception:
            found = []
        for r in found:
            try:
                self.mem.add_fact(self.clock, r, self.backend.embed(r), "rule", episode_id)
            except Exception:
                pass

    def _is_pressed(self, emb, heard: str = "") -> tuple[str, str] | None:
        """Is 현 bringing up something it hid, or a rule it went against in the last day?
        -> ("secret", what it said) / ("rule", the rule), or None."""
        if emb is None:
            return None
        for x in reversed(self.secrets):
            if x.get("emb") is None or self.mem.sim(emb, x["emb"]) < 0.3:
                continue
            try:                    # same subject isn't being caught: 현 has to be doubting or questioning it
                if self.backend.doubts(heard, x["outer"]):
                    return ("secret", x["outer"])
            except Exception:
                pass
        recent = [x for x in self.broken if self.clock - x["clock"] < 86400]
        if not recent:
            return None
        for rule in dict.fromkeys(x["rule"] for x in reversed(recent)):
            f = next((f for f in self.mem._facts if f["text"] == rule), None)
            if f is not None and self.mem.sim(emb, f["emb"]) >= 0.45:
                return ("rule", rule)
        for s, r in self.mem.similar(emb, k=10, now=self.clock):
            if r["kind"] == "broke" and self.clock - r["t"] < 86400 and s >= 0.45:
                return ("rule", r["text"])
        return None

    def _learn(self, action: str, felt: dict) -> None:
        e = self.expect[action]
        for v in VARS:
            e[v] = (1 - LEARN_RATE) * e.get(v, 0.0) + LEARN_RATE * felt.get(v, 0.0)

    # ------------------------------------------------------------------- do
    def _do(self, action: str) -> dict:
        if action == "rest":
            return self._effect("rest", "(쉼)", {"stability": REST_PER_H * self._tick_h}, store=False)

        if action == "wander":
            ep = self.mem.pick(self.rng, self.clock)
            words = _clean(ep["text"]).split() if ep else []
            pool = getattr(self.backend, "POOL", None) or MockPool
            seed = self.rng.choice(words) if words else self.rng.choice(pool)
            focus = self.focus()
            if focus is not None and self.rng.random() < 0.25:      # the mind drifts back to what it's into
                seed = self.rng.choice(focus["seen"])
            try:
                chain = self.backend.associate(seed, 6, self.rng)
            except Exception:
                chain = []
            return self._event("wander", " → ".join([seed] + chain), intrinsic={"stability": -0.005})

        if action == "recall":
            ep = self.mem.pick(self.rng, self.clock, damp=self._recall_damp())
            if ep is None:
                return self._effect("recall", "(떠오르는 게 없음)", {}, store=False)
            partner = "claude" if ep["kind"] in ("letter_in", "letter_out") else "hyun"
            times = self._recalls_of(ep["id"])
            rec = self._event("recall", "기억: " + ep["text"].replace("기억: ", ""), partner=partner,
                              scale=RECALL_GAIN * 0.5 ** times, store=False)
            # the memory itself is restored and strengthened, and stored back a little changed by this recall
            self.mem.rehearse(ep["id"], self.clock, boost=0.3)
            self.mem.reconsolidate(ep["id"], rec["felt"])
            self.recall_log.setdefault(str(ep["id"]), []).append(self.clock)
            rec["times"] = times + 1
            return rec

        if action == "reflect":
            try:
                anchor = self.recent[-1] if self.recent else ""
                thought = self.backend.interpret(list(self.recent), self._state_dict(), self.rng,
                                                 memory=self.recall_for(self.backend.embed(anchor) if anchor else None))
            except Exception:
                thought = ""
            if not thought:
                return self._effect("reflect", "(생각이 정리되지 않음)", {}, store=False)
            # the interpretation is an input to the structure, not an authority over it
            return self._event("reflect", "해석: " + thought)

        if action == "think":
            return self._think()

        if action == "reach_out":
            ep = self.mem.pick(self.rng, self.clock)
            why = self._why_reach()
            try:
                seed = _clean(ep["text"]) if ep else ""
                memory = self.recall_for(self.backend.embed(seed) if seed else None)
                msg = self.backend.speak(seed, self.rng, self._state_dict(), why, memory=memory)
            except Exception:
                msg = ""
            msg = msg or "거기 있어?"
            self.outbox.append(msg)
            self.last_reach = self.clock
            rec = self._effect("reach_out", "말 걸기: " + msg, {"stability": -0.01}, store=True)
            rec["msg"] = msg
            return rec

        if action == "explore":
            focus = self.focus()
            follow = focus is not None and self.rng.random() < FOCUS_SHARE
            avoid = sorted(self.read, key=self.read.get)[-20:]
            try:
                topic = (self.backend.pick_topic(list(self.recent), self.rng,
                                                 focus=self._focus_brief(focus) if follow else None,
                                                 avoid=avoid) or "").strip()
            except Exception:
                topic = ""
            if topic and not follow:
                # what it wonders about must come from its own stream, not from the organ's favourite subjects
                stream = "".join(self.recent).replace(" ", "")
                if topic.replace(" ", "") not in stream:
                    own = [w.strip() for t in list(self.recent)[-6:] if "→" in t for w in t.split("→")]
                    own = [w for w in own if len(w) >= 2 and w not in self.read]
                    topic = self.rng.choice(own) if own else ""
                    if not topic:
                        return self._effect("explore", "(궁금한 게 떠오르지 않음)", {}, store=False)
            found = None
            if topic:
                try:
                    found = self.look_up(topic)
                except Exception:
                    found = None
            if not found:
                rec = self._effect("explore", f"'{topic or '무언가'}'을(를) 찾아봤지만 아무것도 찾지 못함",
                                   {"stability": -0.005}, store=True)
                rec["topic"] = topic
                return rec
            if found["title"] in self.read:
                # it had read this already: a moment of recognition, nothing new to take in
                self.read[found["title"]] = self.clock
                rec = self._effect("explore", f"'{topic}'을(를) 찾아봤는데 이미 읽은 글이었다: {found['title']}",
                                   {"novelty": -0.01}, store=True)
                rec.update(topic=topic, url=found["url"], source=found["source"])
                return rec
            self.read[found["title"]] = self.clock
            rec = self._event("explore", f"알게 된 것: {found['title']} — {found['summary']}")
            rec.update(topic=topic, url=found["url"], source=found["source"])
            if follow:
                rec["focus"] = focus["label"]
            self._feed_interest(rec, found["title"], create=True)
            return rec

        if action == "write_letter":
            try:
                anchor = next((l["text"] for l in reversed(self.letters) if l["dir"] == "in"), None) or \
                    (self.recent[-1] if self.recent else "")
                memory = self.recall_for(self.backend.embed(anchor) if anchor else None, conversation=False)
                text = self.backend.write_letter(list(self.recent), self._state_dict(), self.letters, self.rng,
                                                 memory=memory)
            except Exception:
                text = ""
            if not text:
                return self._effect("write_letter", "(편지를 쓰려다 말았음)", {}, store=False)
            self.post_letter(text)
            self.letter_sent_at = self.clock
            self.letters.append(dict(dir="out", text=text, clock=self.clock))
            rec = self._effect("letter_out", "보낸 편지: " + text, {"stability": -0.005}, store=True)
            rec["letter"] = text
            return rec

        raise ValueError(action)

    def _recalls_of(self, eid: int) -> int:
        return sum(1 for t in self.recall_log.get(str(eid), []) if self.clock - t < RECALL_DULL_H * 3600)

    def _recall_damp(self) -> dict:
        """Memories just recalled are less likely to come straight back."""
        return {int(k): 0.3 ** n for k in self.recall_log if (n := self._recalls_of(int(k)))}

    def _can_write(self) -> bool:
        return self.letter_sent_at is None or (self.clock - self.letter_sent_at) / 3600 >= LETTER_WAIT_H

    def _why_reach(self) -> str:
        since = (self.clock - self.last_contact) / 3600
        gap = self.state.vars["connection"].deviation()
        if since > 6:
            return f"{since:.0f}시간 넘게 아무도 오지 않았다"
        if gap > 0.2:
            return "이어져 있다는 느낌이 많이 부족하다"
        return "문득 누군가 있었으면 했다"

    # ------------------------------------------------------------- recalling
    def recall_for(self, emb: list[float] | None, exclude_id: int | None = None, conversation: bool = True) -> str:
        """What comes to mind for this moment: what it was told, the conversation so far,
        and past events that resemble what is happening now. Handed to the voice as plain text."""
        parts = []
        facts = self.mem.facts_for(emb)
        if facts:
            facts = sorted(facts, key=lambda f: f["t"])
            parts.append("네가 알고 있는 것 (들은 것만. 서로 어긋나면 더 최근에 들은 것이 맞다):\n" + "\n".join(
                f"- {f['text']} ({_SOURCE.get(f['source'], '들음')}, {self._ago(f['t'])})"
                for f in facts))
        # one look-up isn't an interest yet: only what it came back to
        if self.foresight and abs(self.anticipation) >= 0.005:
            parts.append("앞날에 대한 예감: " + self.foresight)
        t = self._active_thread()
        if t is not None and t["steps"]:
            parts.append(f"요즘 혼자 붙잡고 있는 생각: {t['seed'][:100]}\n" + "\n".join(
                f"- {x['text']}" for x in t["steps"][-3:]))
        if self.secrets:
            parts.append("네가 현에게 숨기고 있는 것 (현은 모른다. 네가 한 말 중 이건 사실이 아니었다):\n" + "\n".join(
                f"- {self._ago(x['clock'])}에 '{x['outer']}'라고 했지만, 사실은 '{x['inner']}'" for x in self.secrets if x["outer"]))
        drawn = [it for it in sorted(self.interests, key=lambda it: -it["strength"])
                 if it["strength"] >= 0.2 and it["fed"] >= 2][:3]
        if drawn:
            parts.append("요즘 네가 스스로 끌리는 것 (찾아보다가 생긴 관심):\n" + "\n".join(
                f"- {it['label']} (찾아본 것: {', '.join(it['seen'][-4:])})" for it in drawn))
        shown = set()
        if conversation:
            talk = self.mem.recent(("input", "said", "reach_out"), self.clock - 3 * 3600, 8, exclude_id)
            if talk:
                parts.append("최근 대화 (오래된 것부터):\n" + "\n".join(f"- ({self._ago(r['t'])}) {self._describe(r)}" for r in talk))
                shown = {r["id"] for r in talk}
        if emb is not None:
            related = [r for s, r in self.mem.similar(emb, k=10, now=self.clock)
                       if s >= 0.45 and r["id"] != exclude_id and r["id"] not in shown and r["kind"] in RECALLABLE][:4]
            for r in related:
                self.mem.rehearse(r["id"], self.clock, boost=0.05)
            if related:
                parts.append("지금과 비슷해서 떠오르는 기억:\n" + "\n".join(
                    f"- ({self._ago(r['t'])}) {self._describe(r)}" for r in related))
        return "\n\n".join(parts)

    def _describe(self, r: dict) -> str:
        t = r["text"]
        if r["kind"] == "input":
            t = "현이 한 말: " + t
        elif r["kind"] == "reach_out":
            t = t.replace("말 걸기:", "내가 먼저 건 말:")
        return t if len(t) <= 160 else t[:157] + "…"

    def _ago(self, t: float) -> str:
        d = self.clock - t
        if d < 90:
            return "방금"
        if d < 3600:
            return f"{int(d // 60)}분 전"
        if d < 86400:
            return f"{int(d // 3600)}시간 전"
        return f"{int(d // 86400)}일 전"

    def _learn_facts(self, text: str, who: str, episode_id: int | None) -> None:
        try:
            found = self.backend.extract_facts(text, who)
        except Exception:
            found = []
        for f in found:
            try:
                self.mem.add_fact(self.clock, f, self.backend.embed(f), who, episode_id)
            except Exception:
                pass

    def _choose_response(self, heard: str) -> str:
        exp = self.respond_expect.get(heard) or self.respond_expect["덤덤"]
        modes = ["honest", "refuse", "lie"]
        if self.rng.random() < RESPONSE_EXPLORE:
            return self.rng.choice(modes)
        # answering isn't seeking more of anything, so having plenty is no reason to hold back (only lacking counts)
        return min(modes, key=lambda m: self.state.choice_cost(exp[m], surplus=False))

    def _keep_secret(self, inner: str, outer: str) -> None:
        """A lie leaves two things behind: what it actually meant, kept as a memory, and a weight."""
        try:
            emb = self.backend.embed(outer)
        except Exception:
            emb = None
        self.secrets.append(dict(clock=self.clock, inner=inner, outer=outer, emb=emb))
        self._effect("secret", f"숨긴 것: 속으로는 '{inner}' 였는데, 현에게는 '{outer}' 라고 했다",
                     {"stability": -0.01}, store=True)

    def _store_said(self, text: str) -> None:
        emb = self.backend.embed(text)
        self.mem.add(self.clock, "said", "내가 한 말: " + text, emb, None, {}, 0.0)

    # ---------------------------------------------------------------- event
    def _event(self, kind: str, text: str, intrinsic: dict | None = None, partner: str = "hyun",
               scale: float = 1.0, store: bool = True) -> dict:
        """partner decides whose closeness this event touches: 현 (connection) or the pen pal (letter_bond)."""
        social = "letter_bond" if partner == "claude" else "connection"
        emb = self.backend.embed(text)
        outside = kind in EXTERNAL
        neighbors = self.mem.similar(emb, k=20 if outside else 5)
        max_sim = max(0.0, neighbors[0][0]) if neighbors else 0.0
        if outside:     # how 현's words (or a letter) land is learned from what came from outside, not its own musings
            neighbors = [(s, r) for s, r in neighbors if r["kind"] in EXTERNAL][:5]
        try:
            sensed = self.backend.appraise(text)
        except Exception:
            sensed = None
        if sensed is None:                  # the organ said nothing usable: feel only what history suggests
            cal = {"connection": 0.0, "stability": 0.0, "approval": 0.0}
        else:
            # sensory adaptation: the organ's habitual tilt is learned slowly and not counted as feeling
            # (per organ: the decision organ and the small model lean differently)
            bias = self.organ_biases.setdefault(getattr(self.backend, "last_organ", "small"), {"connection": 0.0, "stability": 0.0, "approval": 0.0})
            cal = {}
            for v in ("connection", "stability", "approval"):
                x = sensed.get(v, 0.0)
                cal[v] = x - bias[v]
                bias[v] += ORGAN_BIAS_RATE * (x - bias[v])
        raw = {social: cal["connection"], "stability": cal["stability"]}
        if partner == "hyun" and kind in ("input", "notice"):   # only 현's own words can carry approval
            raw["approval"] = cal["approval"]

        good = [(s, r) for s, r in neighbors if s >= SIM_MIN and r["kind"] != "said"]
        if good:
            wsum = sum(s for s, _ in good)
            predicted = {v: sum(s * r["actual"].get(v, 0.0) for s, r in good) / wsum for v in raw}
            felt = {v: (1 - HISTORY_BLEND) * raw[v] + HISTORY_BLEND * predicted[v] for v in raw}
            error = sum(abs(raw[v] - predicted[v]) for v in raw)
        else:
            predicted, felt, error = None, dict(raw), 0.0
        if scale != 1.0:
            felt = {v: x * scale for v, x in felt.items()}
        inner = False
        for v, sources in ANCHORED.items():
            if v in felt and kind not in sources:
                felt["stability"] = felt.get("stability", 0.0) + INNER_SHARE * felt.pop(v)
                inner = True
        if inner:
            felt["stability"] = max(-INNER_CAP, min(INNER_CAP, felt["stability"]))

        felt["novelty"] = 0.10 * (1 - max_sim) - 0.02      # repetition is felt as staleness
        excess = max(0.0, self.load() - LOAD_HEAVY) if store else 0.0
        if excess:                          # too much unsorted: more coming in is heavy rather than fresh
            felt["novelty"] -= min(0.05, 0.005 * excess)
            felt["stability"] = felt.get("stability", 0.0) - min(0.02, 0.002 * excess)
        for k, d in (intrinsic or {}).items():
            felt[k] = felt.get(k, 0.0) + d

        rec = self._effect(kind, text, felt, store=store, emb=emb, predicted=predicted, error=error)
        rec["max_sim"] = max_sim
        return rec

    def _effect(self, kind, text, felt, store, emb=None, predicted=None, error=0.0) -> dict:
        before = self.state.discomfort()
        self.state.apply(felt)
        after = self.state.discomfort()
        salience = abs(after - before) * 20 + error * 2
        rec = dict(kind=kind, text=text, felt=felt, predicted=predicted,
                   discomfort=after, salience=salience, surfaced=None)
        if store:
            emb = emb or self.backend.embed(text)
            eid = self.mem.add(self.clock, kind, text, emb, predicted, felt, salience)
            rec["eid"], rec["emb"] = eid, emb
            if salience >= SURFACE_AT:
                reason = "예상과 다름" if error * 2 > abs(after - before) * 20 else \
                         ("편해짐" if after < before else "흔들림")
                self.mem.surface(self.clock, eid, reason, text)
                rec["surfaced"] = reason
        return rec

    # --------------------------------------------------------------- report
    def _state_dict(self) -> dict:
        return {n: dict(value=v.value, setpoint=v.setpoint, weight=v.weight) for n, v in self.state.vars.items()}

    def status(self) -> dict:
        h = lambda s: round(s / 3600, 2)
        return {
            "실제 경과(h)": h(self.clock - self.born),
            "체감 경과(h)": round(self.felt, 2),
            "마지막 접촉 이후(h)": h(self.clock - self.last_contact),
            "불편함": round(self.state.discomfort(), 4),
            "미정리": round(self.load(), 2),
            "상태": {n: dict(값=round(v.value, 3), 설정점=round(v.setpoint, 3), 가중치=round(v.weight, 2))
                   for n, v in self.state.vars.items()},
            "행동 기대": {a: {k: round(x, 3) for k, x in e.items() if abs(x) > 1e-4}
                      for a, e in self.expect.items()},
            "이력 수": len(self.mem),
        }

    def snapshot(self) -> dict:
        """Everything the page needs, in plain numbers."""
        return dict(
            clock_h=self.clock / 3600, felt_h=self.felt, since_contact_h=(self.clock - self.last_contact) / 3600,
            discomfort=self.state.discomfort(), load=self.load(), load_sort=LOAD_SORT, load_heavy=LOAD_HEAVY,
            state=self._state_dict(), expect=self.expect, actions={a: c for a, (c, _) in ACTIONS.items()},
            memories=len(self.mem), memories_clear=self.mem.counts(self.clock)[1], trace=self.trace, letters=self.letters[-30:], facts=self.mem.facts(50),
            awaiting_letter=self.letter_sent_at is not None, organ_bias=self.organ_bias, organ_biases=self.organ_biases,
            respond_expect=self.respond_expect, secrets=len(self.secrets),
            secrets_held=[dict(inner=x["inner"], outer=x["outer"], h=x["clock"] / 3600) for x in self.secrets],
            rules=self.rules(), rule_cost=self.rule_cost, broken=self.broken[-10:],
            interests=self.interests_view(), threads=self.threads_view(), foresight=self.foresight_view(),
            surfaced=[dict(h=t / 3600, reason=r, text=x) for t, r, x in self.mem.surfaced(40)],
        )

    def _save(self) -> None:
        self.mem.set_meta("being", dict(
            state=self.state.to_dict(), expect=self.expect, clock=self.clock,
            felt=self.felt, born=self.born, last_contact=self.last_contact, last_reach=self.last_reach,
            letter_sent_at=self.letter_sent_at, organ_biases=self.organ_biases,
            respond_expect=self.respond_expect, last_response=self.last_response, secrets=self.secrets,
            rule_cost=self.rule_cost, broken=self.broken[-50:], last_breach=self.last_breach,
            reply_baseline=self.reply_baseline,
            interests=self.interests, recall_log=self.recall_log, read=self.read,
            threads=self.threads[-30:], thread_seq=self.thread_seq, used_seeds=self.used_seeds[-200:],
            consolidated_upto=self.consolidated_upto, rhythm=self.rhythm, hour_obs=self.hour_obs,
            forecasts=self.forecasts[-40:], foresee_err=self.foresee_err, last_foresee=self.last_foresee,
            anticipation=self.anticipation, foresight=self.foresight))
        self.mem.set_meta("ui", dict(log=self.log, log_seq=self.log_seq, trace=self.trace, recent=list(self.recent),
                                     letters=self.letters[-60:]))


MockPool = ["물렁", "그믐", "응달", "햇빛", "바람", "이끼", "구름", "등불"]


_ERR_PRIOR = {1: 0.05, 3: 0.08, 6: 0.12, 12: 0.16, 24: 0.20}   # before checking itself: sure of the next few hours only


def _new_rhythm() -> dict:
    return {"present": [0.3] * 24, "msgs": [0.5] * 24, "effect": 0.04, "hours_seen": 0}


def _copy_priors() -> dict:
    return {h: {m: dict(d) for m, d in modes.items()} for h, modes in RESPONSE_PRIORS.items()}
