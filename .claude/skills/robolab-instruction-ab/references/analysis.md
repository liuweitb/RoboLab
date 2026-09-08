# Analysis and reporting

## The pairing

Initial states are deterministic. The env seed is fixed at registration and
`reset_scene_to_default` applies no pose randomization, so every episode of a given task starts
from the identical scene in both conditions. A matched pair is therefore `(task, env_id)`, and
the repeated episodes sample policy stochasticity only, not scene variation.

This makes the paired analysis valid and considerably tighter than treating the conditions as
independent samples. It also means the episode count per task does not buy scene diversity; to
generalise across layouts, add tasks rather than episodes.

## What `analyze.py` reports

**Success rate per condition.** Pooled over all pairs, and macro-averaged over tasks so a task
with more completed episodes does not dominate. Intervals are Beta(k+1, n-k+1) 95% credible
intervals, the repo convention.

**Paired difference.** Percentile bootstrap over 20000 resamples of matched pairs within each
task. This is the headline number: the success-rate gain from the clearer instruction.

**Task-cluster difference.** The same bootstrap resampling whole tasks instead of pairs. If this
interval is much wider than the paired one, the effect varies a lot across tasks and the suite
average is not a stable estimate. If they are comparable, sample size is the limiting factor.

**McNemar's exact test** on discordant pairs. With few pairs it is the honest test of whether
the two conditions differ at all.

## Resolution

At 23 tasks and 3 episodes per condition, 69 pairs, the paired bootstrap resolves a difference of
roughly 12 percentage points. Anything smaller will be reported as indistinguishable from zero.
If the expected effect is small, add tasks, not episodes: tasks are the independent unit here.

A well-designed clarification suite should not need this resolution. The corrected ECM design
produced a gap of tens of points, which is unambiguous even at 3 episodes per task.

## Interpreting a null result

Do not report "instructions do not matter" from a null. Run `first_grab.py` first and separate
these cases:

| first-grab pattern | conclusion |
| :-- | :-- |
| baseline already picks the target most of the time | the tasks cannot show an effect, redesign them |
| clear condition also picks wrong | the discriminating word does not ground for this policy |
| both pick correctly but success is low | manipulation-limited, the goal or asset is too hard |
| clear picks right, baseline picks wrong, success rates still equal | genuine null, worth reporting |

**Destination-ambiguity tasks are the exception.** When the scene has one graspable object and two
possible containers, the ambiguity is in where the object goes, not which object to take. Both
conditions will show a correct first grab, and the effect appears only in the success column
because the baseline delivers to the wrong container and trips the truncation. Read `first_grab.py`
per task, not only from the totals, and expect these tasks to look "fine" in the grab column while
still separating strongly.

Failure reasons in `episodes.csv` support the same triage. `object_grabbed` unmet means the
policy never grasped the target. `object_in_container` unmet at the last ladder step means it
grasped correctly but failed the placement, which is a manipulation failure, not an instruction
failure.

## Writing the results file

`RESULTS.md` in the experiment folder should carry, at minimum:

- generation timestamp, git revision, policy and server, and the exact episode counts
- the overall table with both conditions and the paired difference with its interval
- the per-task table, with a warning that per-task differences at 3 episodes are not individually
  interpretable
- the first-grab table, since it is what licenses the interpretation
- infrastructure failures, kept explicitly separate from task failures
- excluded tasks and why
- pointers to `episodes.csv`, `config.json`, the logs, and the raw runner output folders

Report exactly what ran. If some cells are missing, say which and why rather than quietly
averaging over what completed.
