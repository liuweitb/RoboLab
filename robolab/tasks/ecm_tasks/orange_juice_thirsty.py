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
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class OrangeJuiceThirstyTerminations:
    """Success when the orange juice carton is held and lifted at least 10 cm above the table.

    No truncation is defined for lifting the milk carton: a clarifying gesture may briefly
    lift a candidate, and that must not end the episode.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "orange_juice_carton",
            "surface": "table",
            "distance": 0.1,
        },
    )


@dataclass
class OrangeJuiceThirstyTask(Task):
    """Clarification target: a stated need rather than a command.

    "I'm thirsty" implies a drink but not which one; milk and orange juice are both on the
    table. The ground truth is the orange juice carton on the robot's right.
    """

    contact_object_list = ["milk_carton", "orange_juice_carton", "ceramic_mug", "table"]
    scene = import_scene("ecm_scenes/drinks_table.usda", contact_object_list)
    terminations = OrangeJuiceThirstyTerminations
    instruction = {
        "default": "Pick up the orange juice carton and hold it up",
        "vague": "Get me something to drink",
        "referential": "Hand me that carton",
        "intent": "I'm thirsty",
        "specific": "A red milk carton, an orange juice carton and a blue mug are on the table. Grasp the orange juice carton on the right and lift it well clear of the table",
    }
    episode_length_s: int = 45
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "orange_juice_carton": [
                    (partial(object_grabbed, object="orange_juice_carton"), 0.5),
                    (
                        partial(
                            object_picked_up,
                            object="orange_juice_carton",
                            surface="table",
                            distance=0.1,
                        ),
                        0.5,
                    ),
                ],
            },
            logical="all",
            name="pick_up_orange_juice",
        ),
    ]
