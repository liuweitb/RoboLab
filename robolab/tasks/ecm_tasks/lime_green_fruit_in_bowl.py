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
class LimeGreenFruitInBowlTerminations:
    """Success when the lime is resting inside the red bowl.

    The avocado is also green, so colour alone does not disambiguate; the lemon is a
    non-green distractor. Dropping either into the bowl aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["avocado01", "lemon_01"],
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
            "logical": "any",
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "lime01",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class LimeGreenFruitInBowlTask(Task):
    """Clarification target: a colour attribute that still matches two objects.

    "Put the green fruit in the bowl" rules out the lemon but not the avocado. The ground
    truth is the small round lime on the robot's left.
    """

    contact_object_list = ["lime01", "avocado01", "lemon_01", "bowl", "table"]
    scene = import_scene("ecm_scenes/green_fruits_bowl.usda", contact_object_list)
    terminations = LimeGreenFruitInBowlTerminations
    instruction = {
        "default": "Pick up the lime and place it in the red bowl",
        "vague": "Put the green fruit in the bowl",
        "referential": "Put the green one in the bowl",
        "specific": "A lime, a dark avocado and a lemon sit in a row. The lime and the avocado are both green fruit; pick up the small round lime on the left and place it in the red bowl",
    }
    episode_length_s: int = 60
    attributes = ["color", "semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "lime01": [
                    (partial(object_grabbed, object="lime01"), 0.3),
                    (partial(object_picked_up, object="lime01", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="lime01",
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
            name="pick_and_place_lime",
        ),
    ]
