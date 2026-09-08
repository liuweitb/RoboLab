---
name: robolab-instruction-ab
description: >
  Run a paired A/B evaluation of two instruction variants of the same RoboLab tasks
  (for example oracle-clear vs ambiguous baseline) against a VLA policy, and report the
  success-rate difference with confidence intervals. Use this skill when a user wants to
  measure whether clearer or disambiguated instructions raise policy success rate, wants to
  set up a clarification/ECM headroom experiment, asks why two instruction conditions score
  the same, or asks to rerun such an A/B on a new task suite.
license: CC-BY-NC-4.0
compatibility: >
  Requires a RoboLab checkout with policies/pi0_family/run.py, tasks that define at least two
  instruction variants, and a reachable policy server. Isaac Sim runs locally from .venv.
metadata:
  author: robolab
  version: "1.0.0"
---

# Instruction A/B evaluation

Measure how much a policy's success rate depends on the wording of the instruction, by running
the same tasks twice under two instruction variants with matched initial states.

The canonical use is an **embodied clarification** study: condition A is the oracle-clear
instruction a perfect clarifier would produce, condition B is the ambiguous instruction the
user actually said. The gap between them is the headroom any clarification model can win.
The same machinery works for any two instruction keys defined on the tasks.

## The one thing that ruins this experiment

**A task only measures instruction-following if the policy's default behaviour is wrong.**

Given an ambiguous prompt a VLA does not act randomly. It has a strong prior: it reaches for the
centre object of a row, the larger object, the more salient or more familiar category. If the
task's ground truth happens to coincide with that default pick, the ambiguous baseline succeeds
just as often as the clear instruction and the two conditions tie no matter how good the
clarification is.

Before running, verify for every task that the target is **not** the policy's default pick, and
after any run, verify it from the logs with `scripts/first_grab.py`. See
`references/task_design.md` for the full rule set and the measured evidence behind it.

## Workflow

### 1. Confirm the tasks are suitable

Ask the user which task directory and which two instruction keys to compare, then check each
task defines both keys. Tasks silently fall back to `default` when a key is missing, which
would compare a condition against itself:

```bash
python3 .claude/skills/robolab-instruction-ab/scripts/check_variants.py \
    --tasks-dir robolab/tasks/<dir> --conditions specific referential
```

Report any task missing a key. Either exclude it or add the variant before running.

If prior runs of these tasks exist, run the default-pick diagnostic on them now
(step 5) rather than discovering the tie after a two-hour run.

### 2. Write the experiment config

Copy `assets/config.template.json` into a new experiment folder under `output/` and fill it in.
Every field is recorded so the run is reproducible: task list with both instruction strings, the
policy server, the flags, the git revision, the retry policy.

```bash
EXP=output/<name>_$(date +%F)
mkdir -p $EXP/logs
cp .claude/skills/robolab-instruction-ab/assets/config.template.json $EXP/config.json
cp .claude/skills/robolab-instruction-ab/scripts/{driver.py,analyze.py,status.py,first_grab.py,launch_tmux.sh,watch.sh} $EXP/
```

Set `output_prefix` to something unique. Runner output goes to
`output/<output_prefix>_<condition>_s<session>/`, which must not collide with earlier runs, or
the driver will treat their episodes as already done and skip work.

### 3. Launch

```bash
bash $EXP/launch_tmux.sh all      # two tmux sessions, one per GPU, disjoint blocks
tmux new-session -d -s <name>_watch "bash $EXP/watch.sh"
```

Always run in detached tmux so the experiment survives the session ending. Tell the user the
session names and the expected finish time. Estimate it as:

```
tasks x conditions x (episode_length_s x steps_per_s / steps_per_s) / parallel_envs
```

In practice budget about 4 to 6 minutes per (task, condition) at 3 environments in parallel,
plus 1 to 3 minutes of Kit startup per runner invocation, then halve it for two GPUs.

### 4. Monitor

The watch script prints per-task tallies every 30 s. Interpret the wall column with care: it is
the duration of one runner invocation covering all parallel episodes, so a large value means an
episode ran to timeout and `0` means that condition has not started yet. It is not a quality
signal.

Kit startup crashes with `ERROR_DEVICE_LOST` are common on shared nodes and are **not** task
failures. The driver retries them, keeps them in `infra_errors.jsonl`, and never lets them enter
`episodes.csv`. If a block reports `UNRESOLVED`, rerun `launch_tmux.sh`; completed episodes are
skipped.

### 5. Verify the conditions actually differed

Before believing any result, check what the policy grabbed first in each condition:

```bash
python3 $EXP/first_grab.py --exp $EXP
```

This reads the per-episode event logs and reports, per task and condition, whether the first
grasped object was the target. Two readings matter:

- **Baseline first-grab accuracy is high.** The tasks are not testing disambiguation. Fix the
  task design before drawing conclusions.
- **Clear-condition first-grab accuracy is low.** The discriminating word does not ground for
  this policy, for example a category name for a small lookalike object. Choose a different
  discriminator.

### 6. Analyse and report

```bash
.venv/bin/python $EXP/analyze.py
```

It writes `summary.json`, `summary_per_task.csv`, and prints pooled and macro success rates with
credible intervals, the paired difference with bootstrap intervals, and McNemar's exact test.
See `references/analysis.md` for what each number means and how to word the conclusion.

Write the findings into `$EXP/RESULTS.md` including the excluded tasks, the infra failures, and
the first-grab table. A difference is only reportable alongside the first-grab check, since that
is what distinguishes "instructions did not matter" from "the tasks could not show it".

## Reference files

- `references/task_design.md` — what makes a task able to show an instruction effect, the
  measured default picks of pi05, and which discriminators ground.
- `references/analysis.md` — the statistics, sample-size guidance, failure-mode taxonomy.
- `references/infrastructure.md` — GPU pinning, the Kit startup lock, retry policy, resume
  semantics, and how runner output folders work.

## Scripts

| script | purpose |
| :-- | :-- |
| `check_variants.py` | verify every task defines both instruction keys |
| `driver.py` | orchestrate blocks x conditions, retry infra failures, rebuild `episodes.csv` |
| `launch_tmux.sh` | start one detached driver per GPU with disjoint blocks |
| `status.py` | per-task tallies and episode counts |
| `watch.sh` | live refreshing view for a tmux pane |
| `first_grab.py` | per-condition default-pick diagnostic from event logs |
| `analyze.py` | success rates, paired bootstrap difference, McNemar |

## Worked example

The ECM clarification study on this repo, September 2026:

| run | design | clear | baseline |
| :-- | :-- | :-- | :-- |
| first attempt | target = policy default pick, 2-sentence clear prompts | 49% | 46% |
| pilot after redesign | non-default target, one-line prompts, 10 tasks | 57% | 3% |
| v2 suite, 23 tasks | same recipe applied to the full suite | see `output/ecm_ab_v2_2026-09-07` | |

The first result was reported as "no detectable difference"; the diagnostic showed the baseline
had already been grabbing the right object in 50 of 69 episodes. Nothing about the policy changed
between the runs, only the tasks.
