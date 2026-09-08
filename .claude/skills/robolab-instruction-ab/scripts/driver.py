#!/usr/bin/env python3
"""Orchestrate a paired instruction A/B over a set of RoboLab tasks.

Runs the policy runner once per (block, condition) with ABBA ordering, retries infrastructure
failures, resumes without duplicating episodes (the runner skips runs already present in
output/<prefix>_<cond>_s<session>/episode_results.jsonl), and rebuilds episodes.csv after every
invocation.

Everything task- and experiment-specific comes from config.json next to this file.
"""
import argparse, csv, datetime, fcntl, glob, json, os, subprocess, sys, time

EXP = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(EXP, "config.json")))
REPO = CFG["repo"]
CONDS = [CFG["conditions"]["a"], CFG["conditions"]["b"]]
PREFIX = CFG["output_prefix"]
N_EP = CFG["episodes_per_condition_per_task"]
MAX_ATTEMPTS = CFG.get("max_attempts", 12)
COOLDOWN_S = CFG.get("cooldown_s", 60)
LAST_CRASH = os.path.join(EXP, "last_crash.txt")
CRASH_SIGS = ("ERROR_DEVICE_LOST", "A GPU crash occurred", "Terminated with error", "Segmentation fault")
LOGS = os.path.join(EXP, "logs"); os.makedirs(LOGS, exist_ok=True)
INFRA = os.path.join(EXP, "infra_errors.jsonl")
TASKS = [t["task"] for t in CFG["tasks"]]
BS = CFG.get("block_size", 4)
BLOCKS = [TASKS[i:i + BS] for i in range(0, len(TASKS), BS)]

ap = argparse.ArgumentParser()
ap.add_argument("--session", type=int, default=1, help="session id; runner output goes to output/<prefix>_<cond>_s<id>")
ap.add_argument("--gpu", type=int, default=6, help="Kit/CUDA GPU index (multi-GPU rendering is disabled)")
ap.add_argument("--blocks", type=str, default=None, help="comma-separated block indices for this session (default: all)")
ap.add_argument("--csv-only", action="store_true", help="just rebuild episodes.csv and exit")
ARGS = ap.parse_args()
MY_BLOCKS = [int(b) for b in ARGS.blocks.split(",")] if ARGS.blocks else list(range(len(BLOCKS)))


def out_name(cond):
    return f"{PREFIX}_{cond}_s{ARGS.session}"


def load_results(cond, all_sessions=False):
    pattern = f"{PREFIX}_{cond}_s*" if all_sessions else out_name(cond)
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "output", pattern))):
        p = os.path.join(d, "episode_results.jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    r["_session"] = os.path.basename(d).rsplit("_s", 1)[-1]
                    rows.append(r)
    return rows


def done_episodes(cond, task):
    return {r["episode"] for r in load_results(cond, all_sessions=True) if r["env_name"] == task}


def pending(cond, tasks):
    return [t for t in tasks if len(done_episodes(cond, t)) < N_EP]


def log_infra(rec):
    rec["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(INFRA, "a") as f:
        f.write(json.dumps(rec) + "\n")


def invoke(cond, tasks, attempt, block_idx):
    srv = CFG["policy_server"]
    cmd = [os.path.join(REPO, ".venv/bin/python"), CFG.get("runner", "policies/pi0_family/run.py"),
           "--policy", srv["policy"], "--headless", "--device", f"cuda:{ARGS.gpu}",
           "--kit_args", f"--/renderer/multiGpu/enabled=false --/renderer/activeGpu={ARGS.gpu}",
           "--instruction-type", cond,
           "--num-envs", str(CFG["num_envs"]), "--num-runs", str(CFG["num_runs"]),
           "--remote-host", srv["host"], "--remote-port", str(srv["port"]),
           "--output-folder-name", out_name(cond)]
    if CFG.get("task_dirs"):
        cmd += ["--task-dirs", *CFG["task_dirs"]]
    cmd += ["--task", *tasks]

    stamp = datetime.datetime.now().strftime("%H%M%S")
    log = os.path.join(LOGS, f"s{ARGS.session}_block{block_idx}_{cond}_try{attempt}_{stamp}.log")
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    t0 = time.time()
    # Two Kit instances passing renderer init at the same time crash both with ERROR_DEVICE_LOST,
    # so serialize the startup phase across sessions with a file lock; stepping may overlap.
    lock = open(os.path.join(EXP, "kit_startup.lock"), "w")
    print("[driver]   waiting for startup lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("[driver]   startup lock acquired", flush=True)
    try:
        since = time.time() - float(open(LAST_CRASH).read().strip())
    except Exception:
        since = COOLDOWN_S
    if since < COOLDOWN_S:
        print(f"[driver]   cooling down {COOLDOWN_S - since:.0f}s after the last crash/kill", flush=True)
        time.sleep(COOLDOWN_S - since)
    with open(log, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n"); f.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env)
        held = True
        while True:
            rc = proc.poll()
            if held:
                tail = open(log, errors="ignore").read()
                past_startup = ("[RoboLab] Running " in tail) or ("Connected to" in tail)
                if rc is not None or past_startup or time.time() - t0 > 600:
                    fcntl.flock(lock, fcntl.LOCK_UN); lock.close(); held = False
                    print(f"[driver]   startup lock released after {time.time()-t0:.0f}s", flush=True)
            if rc is not None:
                break
            time.sleep(5)
    text = open(log, errors="ignore").read()
    sigs = [s for s in CRASH_SIGS if s in text]
    return rc, log, time.time() - t0, sigs


def rebuild_csv():
    rows = []
    for cond in CONDS:
        for r in load_results(cond, all_sessions=True):
            rows.append({"task": r["env_name"], "condition": cond, "episode": r["episode"],
                         "env_id": r.get("env_id"), "session": r["_session"], "run": r.get("run"),
                         "success": int(bool(r["success"])), "score": r.get("score"),
                         "episode_step": r.get("episode_step"), "sim_duration_s": r.get("duration"),
                         "wall_total_s": (r.get("timing") or {}).get("wall_total_s"),
                         "it_per_sec": (r.get("timing") or {}).get("it_per_sec"),
                         "reason": r.get("reason"), "instruction": r.get("instruction")})
    rows.sort(key=lambda x: (x["task"], x["condition"], x["episode"]))
    with open(os.path.join(EXP, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task"])
        w.writeheader(); w.writerows(rows)
    return len(rows)


def main():
    start = time.time()
    for bi, block in enumerate(BLOCKS):
        if bi not in MY_BLOCKS:
            continue
        order = CONDS if bi % 2 == 0 else CONDS[::-1]
        for cond in order:
            todo = pending(cond, block)
            if not todo:
                print(f"[driver] block {bi} {cond}: already complete", flush=True); continue
            for attempt in range(1, MAX_ATTEMPTS + 1):
                print(f"[driver] block {bi} {cond} attempt {attempt}: {todo}", flush=True)
                rc, log, wall, sigs = invoke(cond, todo, attempt, bi)
                todo_after = pending(cond, block)
                print(f"[driver]   rc={rc} wall={wall/60:.1f}min sigs={sigs} "
                      f"remaining={todo_after} csv_rows={rebuild_csv()}", flush=True)
                if rc != 0 or sigs:
                    open(LAST_CRASH, "w").write(str(time.time()))
                if rc != 0 or sigs or todo_after:
                    log_infra({"session": ARGS.session, "gpu": ARGS.gpu, "block": bi, "condition": cond,
                               "attempt": attempt, "rc": rc, "signatures": sigs, "tasks_requested": todo,
                               "tasks_still_pending": todo_after, "log": log, "wall_s": round(wall)})
                if not todo_after:
                    break
                todo = todo_after
            else:
                print(f"[driver]   UNRESOLVED block {bi} {cond}: {todo}", flush=True)
    print(f"[driver] finished in {(time.time()-start)/3600:.2f} h; csv rows={rebuild_csv()}", flush=True)


if __name__ == "__main__":
    if ARGS.csv_only:
        print(rebuild_csv(), "rows"); sys.exit(0)
    print(f"[driver] session {ARGS.session} gpu {ARGS.gpu} blocks {MY_BLOCKS} -> "
          f"{[BLOCKS[b] for b in MY_BLOCKS]}", flush=True)
    main()
