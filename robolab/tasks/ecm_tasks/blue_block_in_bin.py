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
class BlueBlockInBinTerminations:
    """Success when the blue block is resting inside the grey bin.

    The red and green blocks are distractors that differ only in colour; dropping either
    into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["red_block", "green_block"],
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
            "object": "blue_block",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BlueBlockInBinTask(Task):
    """Clarification target: three blocks that differ only in colour.

    "Put the block in the bin" omits the colour. The ground truth is the blue block in the
    middle; the clarifier should ask which colour and point at one block.
    """

    contact_object_list = ["red_block", "blue_block", "green_block", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/three_blocks_bin.usda", contact_object_list)
    terminations = BlueBlockInBinTerminations
    instruction = {
        "default": "Pick up the blue block and put it in the bin",
        "vague": "Put the block in the bin",
        "referential": "Put that block in the bin",
        "specific": "Three cubes, red, blue and green, sit in a row in front of the grey bin. Pick up the blue cube in the middle and drop it in the bin",
    }
    episode_length_s: int = 60
    attributes = ["color", "semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "blue_block": [
                    (partial(object_grabbed, object="blue_block"), 0.3),
                    (partial(object_picked_up, object="blue_block", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="blue_block",
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
            name="pick_and_place_blue_block",
        ),
    ]
