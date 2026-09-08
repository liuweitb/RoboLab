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
class BlueBlockInLeftBinTerminations:
    """Success when the blue block is resting inside the left-hand grey bin.

    Dropping it into the right-hand bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "blue_block",
            "container": "bin_right",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "blue_block",
            "container": "bin_left",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BlueBlockInLeftBinTask(Task):
    """Clarification target: one block, two identical grey bins; the ground truth is the bin on the robot's left.

    Under the ambiguous prompt pi05 uses the right-hand bin by default (measured), so the
    baseline fails unless the destination is clarified.
    """

    contact_object_list = ["blue_block", "bin_left", "bin_right", "table"]
    scene = import_scene("ecm_scenes/block_two_bins.usda", contact_object_list)
    terminations = BlueBlockInLeftBinTerminations
    instruction = {
        "default": "Put the blue block in the left bin",
        "vague": "Put the block in the bin",
        "referential": "Put the block in that bin",
        "specific": "Put the blue block in the left bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

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
                            container="bin_left",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_blue_block_bin_left",
        ),
    ]
