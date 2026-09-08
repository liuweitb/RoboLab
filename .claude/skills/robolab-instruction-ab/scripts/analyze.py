#!/usr/bin/env python3
"""Analyze a paired instruction A/B: per-task and pooled SR, paired difference, CIs, McNemar.

Condition A is the "clear"/treatment arm, condition B the baseline; both come from config.json.
Writes summary.json and summary_per_task.csv next to the config and prints the tables.
"""
import csv, json, math, os, random
from collections import defaultdict

try:
    from scipy.stats import beta as _beta, binomtest
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

EXP = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(EXP, "config.json")))
TASKS = [t["task"] for t in CFG["tasks"]]
A, B = CFG["conditions"]["a"], CFG["conditions"]["b"]
NBOOT = CFG.get("bootstrap_reps", 20000)

rows = list(csv.DictReader(open(os.path.join(EXP, "episodes.csv"))))
R = defaultdict(dict)
for r in rows:
    R[(r["task"], r["condition"])][int(r["episode"])] = r


def beta_ci(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    if HAVE_SCIPY:
        return (_beta.ppf(0.025, k + 1, n - k + 1), _beta.ppf(0.975, k + 1, n - k + 1))
    p = k / n; se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def mean(v):
    return sum(v) / len(v) if v else float("nan")


pairs_by_task, per_task = {}, []
for t in TASKS:
    a, b = R.get((t, A), {}), R.get((t, B), {})
    pairs = [(int(a[e]["success"]), int(b[e]["success"])) for e in sorted(set(a) & set(b))]
    pairs_by_task[t] = pairs
    ka, na = sum(int(x["success"]) for x in a.values()), len(a)
    kb, nb = sum(int(x["success"]) for x in b.values()), len(b)
    per_task.append({
        "task": t, f"n_{A}": na, f"k_{A}": ka, f"sr_{A}": ka / na if na else float("nan"),
        f"ci_{A}": beta_ci(ka, na), f"n_{B}": nb, f"k_{B}": kb,
        f"sr_{B}": kb / nb if nb else float("nan"), f"ci_{B}": beta_ci(kb, nb),
        "n_pairs": len(pairs), "diff_pp": 100 * (ka / na - kb / nb) if na and nb else float("nan"),
        "discordant_a_only": sum(1 for x, y in pairs if x and not y),
        "discordant_b_only": sum(1 for x, y in pairs if y and not x),
        f"wall_{A}_s": mean([float(x["wall_total_s"]) for x in a.values() if x["wall_total_s"]]),
        f"wall_{B}_s": mean([float(x["wall_total_s"]) for x in b.values() if x["wall_total_s"]]),
    })

all_pairs = [p for t in TASKS for p in pairs_by_task[t]]
n = len(all_pairs)
ka, kb = sum(x for x, _ in all_pairs), sum(y for _, y in all_pairs)
d_a = sum(1 for x, y in all_pairs if x and not y)
d_b = sum(1 for x, y in all_pairs if y and not x)

rng = random.Random(0)


def boot_paired(reps):
    out = []
    tasks = [t for t in TASKS if pairs_by_task[t]]
    for _ in range(reps):
        s = c = 0
        for t in tasks:                      # resample pairs within each task
            ps = pairs_by_task[t]
            for _ in ps:
                x, y = ps[rng.randrange(len(ps))]
                s += x - y; c += 1
        out.append(100 * s / c if c else float("nan"))
    return out


def boot_cluster(reps):
    out = []
    tasks = [t for t in TASKS if pairs_by_task[t]]
    for _ in range(reps):
        vals = []
        for _ in tasks:                      # resample whole tasks
            ps = pairs_by_task[tasks[rng.randrange(len(tasks))]]
            vals.append(mean([x - y for x, y in ps]))
        out.append(100 * mean(vals))
    return out


def pct(v, q):
    v = sorted(v); i = max(0, min(len(v) - 1, int(q * (len(v) - 1))))
    return v[i]


paired = boot_paired(NBOOT)
cluster = boot_cluster(NBOOT)
macro_a = mean([per_task[i][f"sr_{A}"] for i in range(len(TASKS)) if per_task[i][f"n_{A}"]])
macro_b = mean([per_task[i][f"sr_{B}"] for i in range(len(TASKS)) if per_task[i][f"n_{B}"]])
mcnemar = binomtest(d_a, d_a + d_b, 0.5).pvalue if (HAVE_SCIPY and d_a + d_b) else float("nan")

overall = {
    "n_pairs": n, "conditions": {"a": A, "b": B},
    f"sr_{A}": ka / n if n else float("nan"), f"ci_{A}": beta_ci(ka, n),
    f"sr_{B}": kb / n if n else float("nan"), f"ci_{B}": beta_ci(kb, n),
    "diff_pp": 100 * (ka - kb) / n if n else float("nan"),
    "diff_ci_paired_pp": [pct(paired, 0.025), pct(paired, 0.975)],
    "macro_diff_pp": 100 * (macro_a - macro_b),
    "diff_ci_task_cluster_pp": [pct(cluster, 0.025), pct(cluster, 0.975)],
    "discordant_a_only": d_a, "discordant_b_only": d_b, "mcnemar_exact_p": mcnemar,
    "bootstrap_reps": NBOOT,
}

json.dump({"overall": overall, "per_task": per_task},
          open(os.path.join(EXP, "summary.json"), "w"), indent=1, default=str)
with open(os.path.join(EXP, "summary_per_task.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_task[0].keys())); w.writeheader(); w.writerows(per_task)

print(f"pairs: {n}")
print(f"  {A:14s} SR {ka}/{n} = {overall[f'sr_{A}']:.1%}  95% CI "
      f"[{overall[f'ci_{A}'][0]:.0%}, {overall[f'ci_{A}'][1]:.0%}]")
print(f"  {B:14s} SR {kb}/{n} = {overall[f'sr_{B}']:.1%}  95% CI "
      f"[{overall[f'ci_{B}'][0]:.0%}, {overall[f'ci_{B}'][1]:.0%}]")
print(f"  difference {overall['diff_pp']:+.1f} pp, paired 95% CI "
      f"[{overall['diff_ci_paired_pp'][0]:+.0f}, {overall['diff_ci_paired_pp'][1]:+.0f}]")
print(f"  macro difference {overall['macro_diff_pp']:+.1f} pp, task-cluster 95% CI "
      f"[{overall['diff_ci_task_cluster_pp'][0]:+.0f}, {overall['diff_ci_task_cluster_pp'][1]:+.0f}]")
print(f"  discordant {d_a} {A}-only vs {d_b} {B}-only, McNemar exact p = {mcnemar:.3g}")
print()
w = max(len(t) for t in TASKS)
print(f"{'task':{w}s} {A:>10s} {B:>12s}   diff pp")
for r in per_task:
    print(f"{r['task']:{w}s} {r[f'k_{A}']:>4d}/{r[f'n_{A}']:<5d} {r[f'k_{B}']:>5d}/{r[f'n_{B}']:<6d} "
          f"{r['diff_pp']:+7.0f}")
print("\nwrote summary.json and summary_per_task.csv")
print("Run first_grab.py before interpreting: a null result is only meaningful if the baseline "
      "actually picked the wrong object.")
