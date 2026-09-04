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
    object_next_to,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BananaNextToMugTerminations:
    """Success when the banana rests within 5 cm of the blue mug and has been released.

    The red bowl is the distractor landmark: setting the banana down beside the bowl
    aborts the episode as a truncation. The bowl and mug start 56 cm apart so the two
    conditions cannot both hold.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_next_to,
        params={
            "object": "banana",
            "reference_object": "bowl",
            "dist": 0.05,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_next_to,
        params={
            "object": "banana",
            "reference_object": "ceramic_mug",
            "dist": 0.05,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaNextToMugTask(Task):
    """Clarification target: an unresolved spatial anchor.

    "Put the banana next to it" names no landmark; the bowl and the mug are equally
    plausible. The ground truth is the blue mug on the robot's right.
    """

    contact_object_list = ["banana", "bowl", "ceramic_mug", "table"]
    scene = import_scene("ecm_scenes/banana_bowl_mug_spatial.usda", contact_object_list)
    terminations = BananaNextToMugTerminations
    instruction = {
        "default": "Move the banana next to the blue mug",
        "vague": "Put the banana next to it",
        "referential": "Move the banana over there",
        "specific": "A red bowl sits at the back left and a blue-and-white mug at the back right, with a banana in front. Pick up the banana and set it down right beside the mug, touching or nearly touching it",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "banana": [
                    (partial(object_grabbed, object="banana"), 0.3),
                    (partial(object_picked_up, object="banana", surface="table"), 0.3),
                    (
                        partial(
                            object_next_to,
                            object="banana",
                            reference_object="ceramic_mug",
                            dist=0.05,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="place_banana_next_to_mug",
        ),
    ]
