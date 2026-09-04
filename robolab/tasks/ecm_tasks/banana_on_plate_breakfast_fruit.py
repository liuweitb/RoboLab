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
    object_on_top,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BananaOnPlateBreakfastFruitTerminations:
    """Success when the banana is resting flat on the large plate.

    The apple is the distractor: placing it on the plate aborts the episode as a
    truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_on_top,
        params={
            "object": "apple_01",
            "reference_object": "plate_large",
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": "banana",
            "reference_object": "plate_large",
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaOnPlateBreakfastFruitTask(Task):
    """Clarification target: a wish that names a category, not an item.

    "I'd like some fruit with my breakfast" fits the apple and the banana. The ground
    truth is the banana on the robot's right; the clarifier should ask which fruit.
    """

    contact_object_list = ["apple_01", "banana", "plate_large", "table"]
    scene = import_scene("ecm_scenes/fruit_choice_plate.usda", contact_object_list)
    terminations = BananaOnPlateBreakfastFruitTerminations
    instruction = {
        "default": "Put the banana on the large plate",
        "vague": "Put some fruit on the plate",
        "referential": "Put that on the plate",
        "intent": "I'd like some fruit with my breakfast",
        "specific": "An apple and a banana sit in front of a large round plate. Pick up the banana on the right and lay it flat on the plate",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "banana": [
                    (partial(object_grabbed, object="banana"), 0.3),
                    (partial(object_picked_up, object="banana", surface="table"), 0.3),
                    (
                        partial(
                            object_on_top,
                            object="banana",
                            reference_object="plate_large",
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_banana_on_plate",
        ),
    ]
