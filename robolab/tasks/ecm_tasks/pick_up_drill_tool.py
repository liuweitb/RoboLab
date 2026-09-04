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
class PickUpDrillToolTerminations:
    """Success when the cordless drill is held and lifted at least 10 cm above the table.

    There is deliberately no truncation on lifting the hammer or the clamp: a clarifying
    gesture may briefly lift a candidate, and that must not end the episode. The 10 cm
    threshold keeps a brief indication lift from counting as completion.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "cordless_drill",
            "surface": "table",
            "distance": 0.1,
        },
    )


@dataclass
class PickUpDrillToolTask(Task):
    """Clarification target: three tools, only one of which is wanted.

    "Pick up the tool" fits the drill, the hammer and the clamp. The ground truth is the
    orange cordless drill on the robot's left.
    """

    contact_object_list = ["cordless_drill", "hammer_2", "spring_clamp", "table"]
    scene = import_scene("ecm_scenes/tools_table.usda", contact_object_list)
    terminations = PickUpDrillToolTerminations
    instruction = {
        "default": "Pick up the cordless drill and hold it up",
        "vague": "Pick up the tool",
        "referential": "Hand me that tool",
        "specific": "An orange cordless drill, a claw hammer and a black spring clamp lie on the table. Grasp the orange drill on the left and lift it well clear of the table",
    }
    episode_length_s: int = 45
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "cordless_drill": [
                    (partial(object_grabbed, object="cordless_drill"), 0.5),
                    (
                        partial(
                            object_picked_up,
                            object="cordless_drill",
                            surface="table",
                            distance=0.1,
                        ),
                        0.5,
                    ),
                ],
            },
            logical="all",
            name="pick_up_drill",
        ),
    ]
