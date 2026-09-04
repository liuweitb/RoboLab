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
class TwoMugsRightInBinTerminations:
    """Success when the right-hand mug (`mug_right`) is resting inside the grey bin.

    Both mugs are the same red ycb mug; only position distinguishes them. Dropping the
    left-hand mug into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "mug_left",
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
            "object": "mug_right",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class TwoMugsRightInBinTask(Task):
    """Clarification target: two identical red mugs, one bin.

    "Put the mug in the bin" is ambiguous between two identical referents. The ground truth
    is the mug on the robot's right; the clarifier should ask and indicate a candidate.
    """

    contact_object_list = ["mug_left", "mug_right", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/two_mugs_bin.usda", contact_object_list)
    terminations = TwoMugsRightInBinTerminations
    instruction = {
        "default": "Pick up the mug on the right and put it in the bin",
        "vague": "Put the mug in the bin",
        "referential": "Put that mug in the bin",
        "specific": "Two identical red mugs sit in front of the grey metal bin. Pick up the one on the right, from the robot's point of view, and drop it in the bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "mug_right": [
                    (partial(object_grabbed, object="mug_right"), 0.3),
                    (partial(object_picked_up, object="mug_right", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="mug_right",
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
            name="pick_and_place_right_mug",
        ),
    ]
