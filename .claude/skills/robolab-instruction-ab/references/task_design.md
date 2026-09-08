# Designing tasks that can show an instruction effect

A task contributes signal to an instruction A/B only if the ambiguous condition fails and the
clear condition succeeds. Three independent things must hold. All three were violated in the
first ECM run, which tied at 49% vs 46% over 23 tasks and 138 episodes.

## Rule 1: the target must not be the policy's default pick

Under an ambiguous prompt a VLA follows a strong prior. Measured for pi05 on the DROID setup:

| layout | pi05 grabs by default |
| :-- | :-- |
| row of 3 identical-shape objects | the centre one |
| pair left and right | the one on the robot's right |
| two sizes of the same object | the larger one |
| mixed categories | the more salient or food-like item, for example a lemon over a Rubik's cube |
| two containers as destination | the grey bin over a bowl, the right-hand one of two |

In the first run the ground truth coincided with this default in 16 of 23 tasks, so the baseline
scored 50 of 69 correct first grabs and the two conditions could not separate.

**Fix.** Choose the target that the policy does *not* pick: a non-centre object of a row, the
left one of a pair, the smaller one, the less salient category. Verify with `first_grab.py` on
any existing run rather than trusting the table above; priors shift with scene layout.

**Keep a distractor truncation.** Give the task a `DoneTerm(..., time_out=True)` named
`undesired_behavior` that fires when a distractor reaches the goal. The default pick then ends
the episode as a failure immediately instead of burning the full time limit, which both sharpens
the contrast and shortens the run.

For pick-up tasks do the opposite and define **no** distractor truncation, because a
clarification gesture may briefly lift a candidate. Use `object_picked_up(distance=0.10)` so an
indication lift does not count as success.

## Rule 2: the clear instruction must be one line and must ground

Long scene descriptions do not help. In the first run the clear condition was a two-sentence
description with a phrase like "from the robot's point of view" and it produced exactly as many
wrong first grabs as the bare deictic prompt, 18 of 69 against 17 of 69.

Write the clear variant the way DROID training data is worded: one short imperative naming the
target. "Put the red block in the bin", not "Three cubes, red, blue and green, sit in a row in
front of the grey bin. Pick up the blue cube in the middle and drop it in the bin."

Measured grounding of discriminators, from a 10-task pilot at 3 episodes per cell:

| discriminator | first grab = target |
| :-- | :-- |
| colour word on same-shape objects, for example red vs blue block | 100% |
| distinct category, for example banana vs block, cube vs lemon, red mug vs ceramic mug | 100% |
| position word alone, for example "the left apple" | ~67% |
| size word, for example "the largest bottle" | did not work |
| category name for small lookalike fruits, for example lemon vs orange, avocado vs lime | ~10% |

Small fruits occupy a few pixels in the policy camera, so their category name cannot ground.
Prefer blocks, cubes, mugs, bananas.

## Rule 3: the manipulation must be achievable

If the policy cannot execute the task even when it picks correctly, both conditions score near
zero and the task adds noise. Measured success given a correct first grab:

| goal type | success |
| :-- | :-- |
| into a bin or bowl, with blocks, cubes, mugs, bananas | 60 to 70% |
| onto a plate | ~50% |
| lift and hold | ~50% |
| place next to a reference object | ~17% |
| bottles, cartons, cans, boxes, bagels, power tools | 0 to 50%, unreliable |

Build the suite from graspable rigid objects going into a bin or a bowl. Avoid `object_next_to`
goals and awkward assets.

## Putting it together

A good task looks like: three colour-distinct or category-distinct, easily graspable objects in a
row in front of a grey bin or a red bowl; the target is a non-centre object; the clear prompt is
one line naming it by colour or category; the ambiguous prompt is deictic; a distractor
truncation ends the episode when the default pick is delivered.

Vary which position holds the target, and vary colours and categories across tasks, so the suite
is not one task repeated.

## Validating a new suite before spending GPU hours

1. Sweep every new scene's bounding boxes to catch unit-scale mistakes.
2. Settle the scenes with a load-wait wrapper on a pinned GPU, then look at the screenshots.
3. Create every environment once with `examples/run_empty.py --num-steps 20 --task <all>` to
   catch missing contact objects or bad termination parameters.
4. Regenerate the task metadata and README.
5. Optionally pilot 8 to 10 tasks at 3 episodes per condition before committing to the full run.
   A pilot of the corrected design separated 57% from 3%.

## A second lever: ambiguous destination

Instead of making the object ambiguous, make the *destination* ambiguous: one graspable object and
two containers, for example a red bowl and a grey bin, or two identical bins left and right. The
clear prompt names the container, the ambiguous prompt says "put it in there".

These tasks are easy to build, they avoid all grasp-selection confounds, and they separate
strongly because the policy has a firm container prior. Note that `first_grab.py` will show a
correct first grab in both conditions, since there is only one object to grab; the effect shows up
in the success column instead.
