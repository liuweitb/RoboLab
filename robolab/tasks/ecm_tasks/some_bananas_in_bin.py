# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_in_container,
    pick_and_place,
)
from robolab.core.task.task import Task


@configclass
class SomeBananasInBinTerminations:
    """Success when exactly two of the three bananas are resting in the grey bin.

    `logical="choose"` with `K=2` fires the moment the second banana lands, so the third
    can never be placed inside a successful episode. `require_contact_with` is False
    because the second banana usually comes to rest on the first.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["banana", "banana_01", "banana_02"],
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
            "logical": "choose",
            "K": 2,
        },
    )


@dataclass
class SomeBananasInBinTask(Task):
    """Clarification target: an underspecified quantity.

    "Put some bananas in the bin" does not say how many. The ground truth is two; a good
    clarifier asks for the count and can indicate two candidate bananas.
    """

    contact_object_list = ["banana", "banana_01", "banana_02", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/bananas_3_bin.usda", contact_object_list)
    terminations = SomeBananasInBinTerminations
    instruction = {
        "default": "Put two of the bananas in the bin",
        "vague": "Put some bananas in the bin",
        "referential": "Put a couple of those in the bin",
        "specific": "Three bananas lie in a row in front of the grey bin. Pick up any two of them, one at a time, and drop them in the bin, leaving the third on the table",
    }
    episode_length_s: int = 90
    attributes = ["semantics", "counting"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    # Exactly two of the three bananas count; which two is left to the robot.
    subtasks = [
        pick_and_place(
            object=["banana", "banana_01", "banana_02"],
            container="bin_b03",
            logical="choose",
            K=2,
            score=1.0,
        ),
    ]
