#!/usr/bin/env python3
"""Print progress of the A/B: episode counts per condition, per-task k/n, infra events.

The wall column is the summed wall-clock duration of the runner invocations for that
(task, condition). One invocation covers all parallel episodes, so a large value means an episode
ran to timeout and 0 means that condition has not started yet. It is a duration, not a quality
signal.
"""
import collections, csv, json, os

EXP = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(EXP, "config.json")))
TASKS = [t["task"] for t in CFG["tasks"]]
A, B = CFG["conditions"]["a"], CFG["conditions"]["b"]

p = os.path.join(EXP, "episodes.csv")
rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
by = collections.defaultdict(lambda: [0, 0, {}])
for r in rows:
    k = (r["task"], r["condition"])
    by[k][0] += int(r["success"]); by[k][1] += 1
    by[k][2][r["run"]] = float(r["wall_total_s"] or 0)

need = CFG["episodes_per_condition_per_task"] * len(TASKS)
tot = {c: sum(v[1] for (t, cc), v in by.items() if cc == c) for c in (A, B)}
print(f"episodes: {A} {tot[A]}/{need}, {B} {tot[B]}/{need}")
w = max(len(t) for t in TASKS)
for t in TASKS:
    a, b = by.get((t, A), [0, 0, {}]), by.get((t, B), [0, 0, {}])
    if a[1] or b[1]:
        print(f"  {t:{w}s} {A} {a[0]}/{a[1]}  {B} {b[0]}/{b[1]}   "
              f"run wall(s): {sum(a[2].values()):.0f}/{sum(b[2].values()):.0f}")
ie = os.path.join(EXP, "infra_errors.jsonl")
if os.path.exists(ie):
    print("infra events:", sum(1 for _ in open(ie)), "(retried startup crashes, not task failures)")
