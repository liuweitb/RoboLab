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
class OrangeFruitInBowlTerminations:
    """Success when the orange is resting inside the red bowl.

    The apple and the lemon are distractors of the same category ("fruit"); dropping
    either into the bowl aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["apple_01", "lemon_01"],
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
            "object": "orange_01",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class OrangeFruitInBowlTask(Task):
    """Clarification target: a category word that matches three objects.

    "Put the fruit in the bowl" fits the apple, the orange and the lemon equally well. The
    ground truth is the orange in the middle.
    """

    contact_object_list = ["apple_01", "orange_01", "lemon_01", "bowl", "table"]
    scene = import_scene("ecm_scenes/fruit_trio_bowl.usda", contact_object_list)
    terminations = OrangeFruitInBowlTerminations
    instruction = {
        "default": "Pick up the orange and place it in the red bowl",
        "vague": "Put the fruit in the bowl",
        "referential": "Put that fruit in the bowl",
        "specific": "An apple, an orange and a lemon sit in a row. Pick up the orange in the middle and place it in the red bowl behind them",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "orange_01": [
                    (partial(object_grabbed, object="orange_01"), 0.3),
                    (partial(object_picked_up, object="orange_01", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="orange_01",
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
            name="pick_and_place_orange",
        ),
    ]
