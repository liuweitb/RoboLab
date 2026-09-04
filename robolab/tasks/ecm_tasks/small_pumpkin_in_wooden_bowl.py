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
    object_in_container,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class SmallPumpkinInWoodenBowlTerminations:
    """Success when the small pumpkin is resting inside the wooden bowl.

    The large pumpkin is the distractor: dropping it into the bowl aborts the episode as a
    truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "pumpkinlarge",
            "container": "wooden_bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "pumpkinsmall",
            "container": "wooden_bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class SmallPumpkinInWoodenBowlTask(Task):
    """Clarification target: two pumpkins that differ only in size.

    "Put the pumpkin in the bowl" does not say which size. The ground truth is the small
    pumpkin on the robot's right.
    """

    contact_object_list = ["pumpkinlarge", "pumpkinsmall", "wooden_bowl", "table"]
    scene = import_scene("ecm_scenes/two_pumpkins_wooden_bowl.usda", contact_object_list)
    terminations = SmallPumpkinInWoodenBowlTerminations
    instruction = {
        "default": "Pick up the small pumpkin and put it in the wooden bowl",
        "vague": "Put the pumpkin in the bowl",
        "referential": "Put that pumpkin in the bowl",
        "specific": "Two orange pumpkins, one large and one small, sit in front of a wooden bowl. Pick up the smaller pumpkin, on the right, and place it in the wooden bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "size"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "pumpkinsmall": [
                    (partial(object_grabbed, object="pumpkinsmall"), 0.3),
                    (partial(object_picked_up, object="pumpkinsmall", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="pumpkinsmall",
                            container="wooden_bowl",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_small_pumpkin",
        ),
    ]
