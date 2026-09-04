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
class OrangeInWoodenBowlTerminations:
    """Success when the orange is resting inside the wooden bowl.

    The small red bowl is the distractor destination: dropping the orange there aborts the
    episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "orange_02",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "orange_02",
            "container": "wooden_bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class OrangeInWoodenBowlTask(Task):
    """Clarification target: one orange, two bowls of different kinds.

    "Put the orange in the bowl" fits both bowls. The ground truth is the large wooden bowl
    on the robot's right; the clarifier should ask which bowl and point at one.
    """

    contact_object_list = ["orange_02", "bowl", "wooden_bowl", "table"]
    scene = import_scene("ecm_scenes/orange_two_bowls.usda", contact_object_list)
    terminations = OrangeInWoodenBowlTerminations
    instruction = {
        "default": "Put the orange in the wooden bowl",
        "vague": "Put the orange in the bowl",
        "referential": "Put the orange in there",
        "specific": "A small red bowl and a large wooden bowl sit behind an orange. Pick up the orange and place it in the large wooden bowl on the right",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "orange_02": [
                    (partial(object_grabbed, object="orange_02"), 0.3),
                    (partial(object_picked_up, object="orange_02", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="orange_02",
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
            name="pick_and_place_orange_wooden_bowl",
        ),
    ]
