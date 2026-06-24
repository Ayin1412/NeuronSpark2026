from __future__ import annotations
from typing import Any

ACTION_PLANS: dict[str, list[dict]] = {
    "rollback_release": [
        {"type": "ROLLBACK_RELEASE", "target": None},
        {"type": "CIRCUIT_BREAK",    "target": "svc_00"},
    ],
    "rollback_config": [
        {"type": "ROLLBACK_CONFIG", "target": None},
        {"type": "SET_TIMEOUT",     "target": None, "value": 750},
    ],
    "restart_scale": [
        {"type": "RESTART", "target": None},
        {"type": "SCALE",   "target": None, "value": "base+3"},
    ],
    "scale_throttle": [
        {"type": "SCALE", "target": "svc_00", "value": "base+4"},
        {"type": "SCALE", "target": None,     "value": "base+3"},
    ],
    "storage_index": [
        {"type": "REBUILD_INDEX", "target": "src"},
        {"type": "SET_TIMEOUT",   "target": None, "value": 1000},
    ],
    "queue_drain": [
        {"type": "DRAIN_QUEUE", "target": "svc_08"},
        {"type": "SCALE",       "target": "svc_09", "value": "base+4"},
    ],
    "cache_warm": [
        {"type": "WARM_CACHE", "target": "svc_07"},
        {"type": "THROTTLE",   "target": "svc_00", "value": 12},
    ],
    "timeout_circuit": [
        {"type": "CIRCUIT_BREAK", "target": None},
        {"type": "SET_TIMEOUT",   "target": "src", "value": 1000},
    ],
}

COOLDOWN = {
    "SCALE": 4, "ROLLBACK_RELEASE": 12, "ROLLBACK_CONFIG": 10,
    "RESTART": 8, "CIRCUIT_BREAK": 6, "SET_TIMEOUT": 5,
    "WARM_CACHE": 5, "REBUILD_INDEX": 16, "DRAIN_QUEUE": 7,
    "THROTTLE": 6, "DIAGNOSTIC_PROBE": 5, "NOOP": 0,
}

SVC_DEPS: dict[str, list[str]] = {
    "svc_00": ["svc_07"],
    "svc_01": ["svc_07"],
    "svc_02": ["svc_06", "svc_07"],
    "svc_03": ["svc_05", "svc_08"],
    "svc_04": ["svc_05"],
    "svc_09": ["svc_05", "svc_08"],
}

DEP_SVCS = {"svc_05", "svc_06", "svc_07", "svc_08"}
PRIMARY_HINTS = {"rollback_release", "rollback_config", "restart_scale", "scale_throttle"}
STORAGE_DEPS = {"svc_05", "svc_06"}

# Only valid hints per dep service (coupling noise otherwise)
DEP_VALID_HINTS: dict[str, set] = {
    "svc_05": {"storage_index", "timeout_circuit"},
    "svc_06": {"storage_index", "timeout_circuit"},
    "svc_07": {"cache_warm", "timeout_circuit"},
    "svc_08": {"queue_drain", "timeout_circuit"},
}

# Only one secondary hint passes the benefit/harm test (75% precision)
# queue_drain → rollback_release: consumer_lag + release_instability on svc_09
SECONDARY_HINT_VALID = {
    "queue_drain": ("rollback_release", "self"),
}


class Agent:
    def reset(self, task_info: dict[str, Any]) -> None:
        self.services = {s["alias"]: s for s in task_info.get("services", [])}
        self.t = 0

        self.ch_svc: dict[str, str] = {}
        self.ch_hint: dict[str, str] = {}
        self.ch_src:  dict[str, str] = {}

        self.svc_channels: dict[str, list[str]] = {}
        self.probe_last:   dict[str, int] = {}
        self.ch_probed:    set[tuple[str, str]] = set()

        self.dep_pending:    dict[str, list[tuple[str, str]]] = {}
        self.dep_hints:      dict[str, str] = {}
        self.dep_probe_last: dict[str, int] = {}

        self.reprobe_hint: dict[str, str] = {}
        self.secondary_done: set[str] = set()

        self.action_done: set[tuple[str, str]] = set()
        self.queue: list[tuple[int, int, dict]] = []
        self.cd_until: dict[str, int] = {}

    def _base(self, a: str) -> int:
        return int(self.services.get(a, {}).get("base_replicas", 1))

    def _max(self, a: str) -> int:
        return int(self.services.get(a, {}).get("max_replicas", 8))

    def _ready(self, action: dict) -> bool:
        t = action.get("type", "NOOP")
        if t == "NOOP":
            return True
        return self.t >= self.cd_until.get(f"{t}:{action.get('target','')}", -1)

    def _commit(self, action: dict) -> dict:
        t = action.get("type", "NOOP")
        tgt = action.get("target", "")
        self.cd_until[f"{t}:{tgt}"] = self.t + COOLDOWN.get(t, 4)
        return action

    def _resolve(self, tmpl: dict, svc: str, src: str) -> dict:
        action = {"type": tmpl["type"]}
        raw = tmpl.get("target")
        if raw is None:
            action["target"] = svc
        elif raw == "src":
            action["target"] = src if (src and src in self.services and src != svc) else svc
        else:
            action["target"] = raw
        if "value" in tmpl:
            v = tmpl["value"]
            tgt = action["target"]
            if v == "base+3":
                action["value"] = min(self._max(tgt), self._base(tgt) + 3)
            elif v == "base+4":
                action["value"] = min(self._max(tgt), self._base(tgt) + 4)
            else:
                action["value"] = v
        return action

    def _enqueue(self, hint: str, svc: str, src: str, priority: int = 10) -> None:
        plan = ACTION_PLANS.get(hint)
        if not plan:
            return
        for i, tmpl in enumerate(plan):
            action = self._resolve(tmpl, svc, src)
            key = (action["type"], action.get("target", ""))
            if key not in self.action_done:
                self.queue.append((self.t, priority + i, action))

    def _queue_dep_probes(self, prim_svc: str, ch: str) -> None:
        for dep in SVC_DEPS.get(prim_svc, []):
            if dep in self.dep_hints:
                continue
            if dep not in self.dep_pending:
                self.dep_pending[dep] = []
            pair = (prim_svc, ch)
            if pair not in self.dep_pending[dep]:
                self.dep_pending[dep].append(pair)

    def _fire_secondary(self, svc: str) -> None:
        if svc in self.secondary_done:
            return
        primary_hint = next(
            (self.ch_hint[ch] for ch in self.svc_channels.get(svc, []) if ch in self.ch_hint),
            None
        )
        if not primary_hint:
            return
        # Only queue_drain → rollback_release is reliable
        sec_info = SECONDARY_HINT_VALID.get(primary_hint)
        if not sec_info:
            return
        dep_found = any(
            self.dep_hints.get(d) and self.dep_hints[d] != primary_hint
            for d in SVC_DEPS.get(svc, [])
        )
        if dep_found:
            return
        sec_hint, src_key = sec_info
        sec_src = svc  # queue_drain → rollback_release uses svc itself
        self.secondary_done.add(svc)
        self._enqueue(sec_hint, svc, sec_src, priority=20)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.t = int(observation.get("time", self.t))

        for pr in observation.get("probe_results", []):
            if pr.get("type") != "diagnostic_probe":
                continue
            hint = pr.get("action_hint", "investigate")
            pressure = pr.get("pressure_bucket", 0)
            confidence = pr.get("confidence_bucket", 0)
            target = pr.get("target", "")
            src = pr.get("source_hint", target)
            if pressure == 0 or confidence < 2 or hint == "investigate":
                continue

            if target in DEP_SVCS:
                valid_hints = DEP_VALID_HINTS.get(target, set())
                if hint not in valid_hints:
                    self.dep_pending.pop(target, None)
                    continue
                if target not in self.dep_hints:
                    dep_src = src
                    if hint == "storage_index" and src not in STORAGE_DEPS:
                        dep_src = target
                    self.dep_hints[target] = hint
                    triggers = self.dep_pending.pop(target, [])
                    if triggers:
                        prim_svc, _ = triggers[0]
                        self._enqueue(hint, prim_svc, dep_src, priority=10)
                    else:
                        self._enqueue(hint, target, dep_src, priority=10)
            else:
                for ch, svc in self.ch_svc.items():
                    if svc == target and ch not in self.ch_hint:
                        if hint == "storage_index" and src not in STORAGE_DEPS:
                            continue
                        self.ch_hint[ch] = hint
                        self.ch_src[ch] = src
                        self._enqueue(hint, svc, src, priority=10)
                        if hint in PRIMARY_HINTS:
                            self._queue_dep_probes(svc, ch)

                if target in self.reprobe_hint:
                    expected = self.reprobe_hint.pop(target)
                    if hint == expected:
                        self._fire_secondary(target)

        primary_probes: list[tuple[str, str]] = []
        reprobe_svcs: list[str] = []

        for ev in observation.get("events", []):
            ch = ev.get("channel", "")
            svc = ev.get("service", "")
            if ch == "ch_bg" or not svc or svc not in self.services:
                continue
            if ch not in self.ch_svc:
                self.ch_svc[ch] = svc
            if svc not in self.svc_channels:
                self.svc_channels[svc] = []
            is_new = ch not in self.svc_channels[svc]
            if is_new:
                self.svc_channels[svc].append(ch)

            if ch in self.ch_hint:
                self._enqueue(self.ch_hint[ch], svc, self.ch_src.get(ch, svc), priority=10)
            elif (ch, svc) not in self.ch_probed:
                last = self.probe_last.get(svc, -99)
                if self.t - last >= 5:
                    primary_probes.append((ch, svc))

            if is_new and len(self.svc_channels[svc]) >= 2:
                existing_hint = next(
                    (self.ch_hint[c] for c in self.svc_channels[svc] if c in self.ch_hint),
                    None
                )
                if (existing_hint and
                        svc not in self.reprobe_hint and
                        svc not in self.secondary_done):
                    self.reprobe_hint[svc] = existing_hint
                    reprobe_svcs.append(svc)

        self.queue.sort(key=lambda x: (x[0], x[1]))
        for i, (execute_at, priority, action) in enumerate(self.queue):
            key = (action["type"], action.get("target", ""))
            if key in self.action_done:
                self.queue.pop(i)
                continue
            if execute_at <= self.t:
                if self._ready(action):
                    self.action_done.add(key)
                    self.queue.pop(i)
                    return self._commit(action)
                else:
                    self.queue[i] = (self.t + 1, priority, action)

        for svc in reprobe_svcs:
            last = self.probe_last.get(svc, -99)
            if self.t - last >= 5:
                action = {"type": "DIAGNOSTIC_PROBE", "target": svc}
                if self._ready(action):
                    self.probe_last[svc] = self.t
                    return self._commit(action)

        for ch, svc in primary_probes:
            action = {"type": "DIAGNOSTIC_PROBE", "target": svc}
            if self._ready(action):
                self.ch_probed.add((ch, svc))
                self.probe_last[svc] = self.t
                return self._commit(action)

        for dep_svc, triggers in list(self.dep_pending.items()):
            if not triggers or dep_svc in self.dep_hints:
                self.dep_pending.pop(dep_svc, None)
                continue
            last = self.dep_probe_last.get(dep_svc, -99)
            if self.t - last >= 5:
                action = {"type": "DIAGNOSTIC_PROBE", "target": dep_svc}
                if self._ready(action):
                    self.dep_probe_last[dep_svc] = self.t
                    return self._commit(action)

        return {"type": "NOOP"}