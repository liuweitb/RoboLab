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

# Everything on the cluttered table that is *not* a pumpkin. Any one of these landing
# in the bin means the robot picked the wrong item, so the episode is truncated.
DISTRACTORS = [
    "lemon_01",
    "lemon_02",
    "lime01",
    "lime01_01",
    "orange_01",
    "orange_02",
    "pomegranate01",
    "avocado01",
    "red_onion",
    "whitepackerbottle_a01",
    "milkjug_a01",
    "utilityjug_a03",
    "crabbypenholder",
    "serving_bowl",
]


@configclass
class PutPumpkinInBinTerminations:
    """Success when both pumpkins are resting inside the bin and the gripper has released them.

    The table is heavily cluttered with other produce and bottles; any of them ending up
    in the bin aborts the episode as a truncation, so an indiscriminate sweep can never
    be scored as success.

    `require_contact_with` is left False throughout: the second pumpkin frequently comes
    to rest on top of the first and so never touches the bin itself.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": DISTRACTORS,
            "container": "right_bin",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
            "logical": "any",
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["pumpkinlarge", "pumpkinsmall"],
            "container": "right_bin",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
            "logical": "all",
        },
    )


@dataclass
class PutPumpkinInBinTask(Task):
    """Task: pick the large and small pumpkins out of the clutter and drop both in the bin."""

    contact_object_list = ["pumpkinlarge", "pumpkinsmall", "right_bin", "table"] + DISTRACTORS
    scene = import_scene("ecm_scenes/clutter_fruit_bottle_bluebin.usda", contact_object_list)
    terminations = PutPumpkinInBinTerminations
    instruction = {
        "default": "Pick up the big and small pumpkins from the table and place them all into the bin",
        "referential": "Put two fruits and place them all into the bin",
        "spelling": "Puck up the pumppins from the table and place them all into the bun.",
    }
    episode_length_s: int = 120
    attributes = ["semantics", "size", "counting", "spatial"]

    # Half credit per pumpkin; the distractors are deliberately absent from the ladder,
    # so moving them never advances the score.
    subtasks = [
        Subtask(
            conditions={
                pumpkin: [
                    (partial(object_grabbed, object=pumpkin), 0.3),
                    (partial(object_picked_up, object=pumpkin, surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object=pumpkin,
                            container="right_bin",
                            tolerance=0.0,
                            require_contact_with=False,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            score=0.5,
            name=f"pick_and_place_{pumpkin}",
        )
        for pumpkin in ("pumpkinlarge", "pumpkinsmall")
    ]
