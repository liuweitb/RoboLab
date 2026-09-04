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
class TwoApplesLeftInBowlTerminations:
    """Success when the left-hand apple (`apple_left`) is resting inside the red bowl.

    The two apples are the same asset, so nothing but position tells them apart. Dropping
    the right-hand apple into the bowl aborts the episode as a truncation, so guessing the
    wrong one can never be scored as success.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "apple_right",
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
            "object": "apple_left",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class TwoApplesLeftInBowlTask(Task):
    """Clarification target: two identical apples, one bowl.

    "Put the apple in the bowl" does not say which apple. A good clarifier asks which one
    and points at (or touches) a candidate; the ground truth is the apple on the robot's
    left.
    """

    contact_object_list = ["apple_left", "apple_right", "bowl", "table"]
    scene = import_scene("ecm_scenes/two_apples_bowl.usda", contact_object_list)
    terminations = TwoApplesLeftInBowlTerminations
    instruction = {
        "default": "Pick up the apple on the left and place it in the red bowl",
        "vague": "Put the apple in the bowl",
        "referential": "Put that apple in the bowl",
        "specific": "Two identical apples sit in front of the red bowl. Pick up the one on the left, from the robot's point of view, and place it inside the bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "apple_left": [
                    (partial(object_grabbed, object="apple_left"), 0.3),
                    (partial(object_picked_up, object="apple_left", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="apple_left",
                            container="bowl",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_left_apple",
        ),
    ]
