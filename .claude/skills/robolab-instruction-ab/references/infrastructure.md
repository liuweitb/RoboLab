# Running the experiment reliably

## Environment

Isaac Sim runs locally from the repo `.venv`. Never set `CUDA_VISIBLE_DEVICES`; select the GPU
with the runner flags instead, and pin Kit's renderer to the same device:

```bash
.venv/bin/python policies/pi0_family/run.py --policy pi05 --headless --device cuda:6 \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=6" \
  --instruction-type specific --num-envs 3 --num-runs 1 \
  --remote-host <server> --remote-port 8001 \
  --output-folder-name <prefix>_specific_s1 --task TaskA TaskB
```

Camera-enabled runs need the multi-GPU renderer disabled or Kit dies with `ERROR_DEVICE_LOST`.
Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the driver environment so device indices match nvidia-smi.

Tasks outside the repo tree can be used without touching the repo by passing an absolute path to
`--task-dirs`, which is the clean way to pilot a design before committing task files.

## Kit startup crashes

On a shared node, two Kit processes passing renderer init at the same time crash both with
`ERROR_DEVICE_LOST`, typically about 140 seconds in and before any episode runs. Crash rates of
half the startups have been observed, and one GPU can be much worse than another on a given
night.

The driver handles this with three mechanisms:

1. **A startup lock.** A file lock held from launch until the runner prints its first
   `[RoboLab] Running` line, so only one Kit is in its startup phase at a time across sessions.
   Steady-state stepping may overlap freely.
2. **A global cool-down.** After any crash or kill, no new Kit start for 60 seconds, enforced
   while holding the lock.
3. **Retries.** Up to 12 attempts per (block, condition). The runner skips episodes already
   present in its output folder, so a retry resumes rather than duplicating.

Crashed invocations are appended to `infra_errors.jsonl` with the session, GPU, block, condition,
attempt, signatures and log path. They never enter `episodes.csv`. Keeping this boundary sharp is
what lets you say "12 startup crashes, no episode lost" instead of contaminating the success rate.

If a GPU crashes repeatedly, move that session's remaining blocks to the other GPU rather than
burning an hour on retries.

## Resume semantics

The runner skips a task whose episodes are already in
`output/<prefix>_<condition>_s<session>/episode_results.jsonl`. Two consequences:

- Rerunning `launch_tmux.sh` after a crash is safe and picks up where it stopped.
- Reusing an `output_prefix` from an earlier experiment makes the driver think work is done and
  silently skip it. Always choose a fresh prefix.

## Block structure and condition order

Tasks are split into blocks of 4. Even blocks run condition A first, odd blocks run condition B
first. This ABBA ordering spreads any drift in server latency or node load across both
conditions rather than letting it favour whichever ran first.

Sessions take disjoint blocks, one per GPU, so no two sessions ever contend for the same task.

## Throughput

With 3 parallel environments the sim steps at roughly 2.6 to 3 iterations per second. A 60 s
episode is 900 steps, so a full-length run of 3 parallel episodes costs about 5 to 6 minutes,
plus about 1 minute of environment creation per task and 1 to 3 minutes of Kit startup per
invocation. A 23-task, 2-condition experiment takes about 2 hours per GPU with two GPUs in
parallel.

Runs finish faster than that when a distractor truncation ends episodes early, which is another
reason to define one.

## Output layout

```
output/<exp_folder>/            config.json driver.py analyze.py status.py first_grab.py
                                launch_tmux.sh watch.sh RUNBOOK.md RESULTS.md
                                episodes.csv infra_errors.jsonl summary.json logs/
output/<prefix>_<cond>_s<N>/    raw runner output: videos, log_<run>_env<id>.json, hdf5,
                                episode_results.jsonl, env_cfg.json
```

`episodes.csv` is rebuilt from the raw jsonl files after every invocation, so it is always
current and can be regenerated at any time with `driver.py --csv-only`.

The per-episode `log_<run>_env<id>.json` files carry the event stream including
`OBJECT_GRABBED_SUCCESS` and `WRONG_OBJECT_GRABBED_FAILURE`, which is what `first_grab.py` reads.
Keep these folders; the diagnostic is impossible without them.
