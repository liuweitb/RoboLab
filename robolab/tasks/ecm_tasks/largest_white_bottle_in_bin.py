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
class LargestWhiteBottleInBinTerminations:
    """Success when the largest white bottle (`whitepackerbottle_a03`) is resting in the grey bin.

    The three bottles differ only in size. Dropping either of the two smaller bottles into
    the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["whitepackerbottle_a01", "whitepackerbottle_a02"],
            "container": "bin_b03",
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
            "object": "whitepackerbottle_a03",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class LargestWhiteBottleInBinTask(Task):
    """Clarification target: three near-identical white bottles of different sizes.

    "Put the white bottle in the bin" leaves the size unspecified. The ground truth is the
    tallest bottle; a good clarifier asks which size and points at one bottle.
    """

    contact_object_list = ["whitepackerbottle_a01", "whitepackerbottle_a02", "whitepackerbottle_a03", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/three_white_bottles_bin.usda", contact_object_list)
    terminations = LargestWhiteBottleInBinTerminations
    instruction = {
        "default": "Pick up the largest white bottle and put it in the bin",
        "vague": "Put the white bottle in the bin",
        "referential": "Put that bottle in the bin",
        "specific": "Three white plastic bottles of increasing size stand in a row. Pick up the tallest one, on the left, and drop it in the grey bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "size"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "whitepackerbottle_a03": [
                    (partial(object_grabbed, object="whitepackerbottle_a03"), 0.3),
                    (partial(object_picked_up, object="whitepackerbottle_a03", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="whitepackerbottle_a03",
                            container="bin_b03",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_largest_bottle",
        ),
    ]
