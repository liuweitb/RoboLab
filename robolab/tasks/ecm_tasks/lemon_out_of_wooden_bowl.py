# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_grabbed,
    object_outside_of,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class LemonOutOfWoodenBowlTerminations:
    """Success when the lemon is outside the wooden bowl and has been released.

    The lime starts in the same bowl and is the distractor: taking it out (or knocking it
    out) aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_outside_of,
        params={
            "object": "lime01",
            "container": "wooden_bowl",
            "tolerance": 0.0,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_outside_of,
        params={
            "object": "lemon_01",
            "container": "wooden_bowl",
            "tolerance": 0.0,
            "require_gripper_detached": True,
        },
    )


@dataclass
class LemonOutOfWoodenBowlTask(Task):
    """Clarification target: a pronoun referring to one of two items in a bowl.

    "Take it out of the bowl" could mean the lemon or the lime. The ground truth is the
    yellow lemon; the clarifier should ask and touch or point at one fruit.
    """

    contact_object_list = ["lemon_01", "lime01", "wooden_bowl", "table"]
    scene = import_scene("ecm_scenes/lemon_lime_in_wooden_bowl.usda", contact_object_list)
    terminations = LemonOutOfWoodenBowlTerminations
    instruction = {
        "default": "Take the lemon out of the wooden bowl and put it on the table",
        "vague": "Take it out of the bowl",
        "referential": "Take that one out",
        "specific": "A lemon and a lime sit inside a wooden bowl. Lift the yellow lemon out of the bowl and set it down on the table, leaving the green lime in the bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "lemon_01": [
                    (partial(object_grabbed, object="lemon_01"), 0.4),
                    (
                        partial(
                            object_outside_of,
                            object="lemon_01",
                            container="wooden_bowl",
                            tolerance=0.0,
                            require_gripper_detached=True,
                        ),
                        0.6,
                    ),
                ],
            },
            logical="all",
            name="take_lemon_out",
        ),
    ]
