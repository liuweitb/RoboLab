#!/usr/bin/env python3
"""Default-pick diagnostic: what did the policy grab FIRST, per task and condition?

This is the check that distinguishes "the instruction did not matter" from "the task could not
show it". It reads the per-episode event logs written by the runner
(output/<prefix>_<cond>_s<N>/<Task>/log_<run>_env<id>.json) and classifies the first grasp.

Read it like this:
  - baseline first-grab accuracy high  -> the target IS the policy's default pick; redesign tasks
  - clear-condition accuracy low       -> the discriminating word does not ground for this policy
  - both high, success low             -> manipulation-limited, not instruction-limited
"""
import argparse, collections, csv, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.dirname(os.path.abspath(__file__)),
                help="experiment folder holding config.json and episodes.csv")
ap.add_argument("--csv", default=None, help="also write the per-task table here")
A = ap.parse_args()

CFG = json.load(open(os.path.join(A.exp, "config.json")))
REPO, PREFIX = CFG["repo"], CFG["output_prefix"]
CONDS = [CFG["conditions"]["a"], CFG["conditions"]["b"]]
TASKS = [t["task"] for t in CFG["tasks"]]

ep_path = os.path.join(A.exp, "episodes.csv")
success = {}
if os.path.exists(ep_path):
    for r in csv.DictReader(open(ep_path)):
        success[(r["task"], r["condition"], str(r["env_id"]), str(r["run"]))] = int(r["success"])


def first_grab(log):
    """'TARGET', the name of the wrongly grabbed object, or 'none'."""
    for e in log.get("events", []):
        if e["name"] == "WRONG_OBJECT_GRABBED_FAILURE":
            m = re.search(r"Wrong object grabbed: '([^']+)'", e["info"])
            return m.group(1) if m else "wrong"
        if e["name"] == "OBJECT_GRABBED_SUCCESS":
            return "TARGET"
    return "none"


res = collections.defaultdict(list)
for cond in CONDS:
    for d in sorted(glob.glob(os.path.join(REPO, "output", f"{PREFIX}_{cond}_s*"))):
        for f in sorted(glob.glob(os.path.join(d, "*", "log_*_env*.json"))):
            try:
                log = json.load(open(f))
            except Exception:
                continue
            task, env = log["task"], str(log["env_id"])
            run = str(log.get("run", 0))
            res[(task, cond)].append((first_grab(log), success.get((task, cond, env, run))))

rows, tot = [], collections.defaultdict(lambda: [0, 0, 0])
w = max([len(t) for t in TASKS] + [10])
print(f"{'task':{w}s} {'condition':12s}  n  first-grab=target  success  first grabs")
for t in TASKS:
    for cond in CONDS:
        k = res.get((t, cond), [])
        if not k:
            print(f"{t:{w}s} {cond:12s}  (no data)"); continue
        g = sum(x[0] == "TARGET" for x in k)
        s = sum(1 for x in k if x[1] == 1)
        tot[cond][0] += g; tot[cond][1] += s; tot[cond][2] += len(k)
        grabs = ", ".join(x[0] for x in k)
        print(f"{t:{w}s} {cond:12s} {len(k):2d}  {g}/{len(k):<15d} {s}/{len(k):<7d} {grabs}")
        rows.append({"task": t, "condition": cond, "n": len(k), "first_grab_target": g,
                     "successes": s, "first_grabs": grabs})
print()
for cond in CONDS:
    g, s, n = tot[cond]
    if n:
        print(f"TOTAL {cond:12s} first-grab=target {g}/{n} = {g/n:.0%}   success {s}/{n} = {s/n:.0%}")

if A.csv and rows:
    with open(A.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {A.csv}")
