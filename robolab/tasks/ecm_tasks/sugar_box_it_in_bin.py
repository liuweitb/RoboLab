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
class SugarBoxItInBinTerminations:
    """Success when the sugar box is resting inside the grey bin.

    The Spam can is the distractor: dropping it into the bin aborts the episode as a
    truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "spam_can",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "sugar_box",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class SugarBoxItInBinTask(Task):
    """Clarification target: a bare pronoun with two very different candidates.

    "Put it in the bin" gives no clue whether "it" is the sugar box or the can. The ground
    truth is the yellow sugar box on the robot's left.
    """

    contact_object_list = ["sugar_box", "spam_can", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/sugar_box_spam_bin.usda", contact_object_list)
    terminations = SugarBoxItInBinTerminations
    instruction = {
        "default": "Put the sugar box in the bin",
        "vague": "Put it in the bin",
        "referential": "Put that in the bin",
        "specific": "A yellow Domino sugar box and a blue Spam can sit in front of the grey bin. Pick up the sugar box on the left and drop it in the bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "sugar_box": [
                    (partial(object_grabbed, object="sugar_box"), 0.3),
                    (partial(object_picked_up, object="sugar_box", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="sugar_box",
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
            name="pick_and_place_sugar_box",
        ),
    ]
