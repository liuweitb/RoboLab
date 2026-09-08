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
class RedBlockInBinTrioTerminations:
    """Success when the red block (right of the row) is resting inside the grey bin.

    Dropping the cube or the banana into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["rubiks_cube", "banana"],
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
            "object": "red_block",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class RedBlockInBinTrioTask(Task):
    """Clarification target: three unrelated objects (cube, banana, block); the ground truth is the red block.

    Under the ambiguous prompt pi05 grabs the centre banana by default (measured), so the
    baseline fails unless the ambiguity is resolved; the one-line specific prompt names the
    target by category.
    """

    contact_object_list = ["rubiks_cube", "banana", "red_block", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/cube_banana_block_bin.usda", contact_object_list)
    terminations = RedBlockInBinTrioTerminations
    instruction = {
        "default": "Put the red block in the bin",
        "vague": "Put the small thing in the bin",
        "referential": "Put that in the bin",
        "specific": "Put the red block in the bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "color"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "red_block": [
                    (partial(object_grabbed, object="red_block"), 0.3),
                    (partial(object_picked_up, object="red_block", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="red_block",
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
            name="pick_and_place_red_block",
        ),
    ]
