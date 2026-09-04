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
class YellowBlockInBinTerminations:
    """Success when the yellow block is resting inside the grey bin.

    The banana and the mustard bottle are also yellow and the red block is a colour foil;
    dropping any of them into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["banana", "mustard", "red_block"],
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
            "object": "yellow_block",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class YellowBlockInBinTask(Task):
    """Clarification target: a colour word shared by three objects.

    "Put the yellow one in the bin" matches the banana, the yellow block and the mustard
    bottle. The ground truth is the yellow block in the middle.
    """

    contact_object_list = ["banana", "yellow_block", "mustard", "red_block", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/yellow_items_bin.usda", contact_object_list)
    terminations = YellowBlockInBinTerminations
    instruction = {
        "default": "Pick up the yellow block and put it in the bin",
        "vague": "Put the yellow one in the bin",
        "referential": "Put the yellow thing in the bin",
        "specific": "A banana, a yellow cube and a yellow mustard bottle are all yellow; a red cube is the odd one out. Pick up the yellow cube in the middle and drop it in the grey bin",
    }
    episode_length_s: int = 60
    attributes = ["color", "semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "yellow_block": [
                    (partial(object_grabbed, object="yellow_block"), 0.3),
                    (partial(object_picked_up, object="yellow_block", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="yellow_block",
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
            name="pick_and_place_yellow_block",
        ),
    ]
