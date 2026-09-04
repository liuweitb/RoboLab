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
    object_on_top,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BagelOnPlateDishTerminations:
    """Success when the bagel is resting flat on the large plate.

    The red bowl is the distractor destination: dropping the bagel into it aborts the
    episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "bagel_00",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": "bagel_00",
            "reference_object": "plate_large",
            "require_gripper_detached": True,
        },
    )


@dataclass
class BagelOnPlateDishTask(Task):
    """Clarification target: the word "dish" covers both a plate and a bowl.

    "Put the bagel on the dish" is ambiguous between the plate and the bowl. The ground
    truth is the large plate on the robot's right.
    """

    contact_object_list = ["bagel_00", "plate_large", "bowl", "table"]
    scene = import_scene("ecm_scenes/bagel_plate_bowl.usda", contact_object_list)
    terminations = BagelOnPlateDishTerminations
    instruction = {
        "default": "Put the bagel on the large plate",
        "vague": "Put the bagel on the dish",
        "referential": "Put the bagel on that",
        "specific": "A large round plate and a red bowl sit behind a bagel. Pick up the bagel and lay it flat on the plate on the right, not in the bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "bagel_00": [
                    (partial(object_grabbed, object="bagel_00"), 0.3),
                    (partial(object_picked_up, object="bagel_00", surface="table"), 0.3),
                    (
                        partial(
                            object_on_top,
                            object="bagel_00",
                            reference_object="plate_large",
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_bagel_on_plate",
        ),
    ]
