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
class LeftBananaInBinTerminations:
    """Success when the left-hand banana (`banana`) is resting inside the grey bin.

    Dropping either other banana into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["banana_01", "banana_02"],
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
            "object": "banana",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class LeftBananaInBinTask(Task):
    """Clarification target: three identical bananas; the ground truth is the one on the robot's left.

    Under the ambiguous prompt pi05 grabs a centre or right-hand banana by default (measured), so the
    baseline fails unless the ambiguity is resolved; the one-line specific prompt names the
    target by position.
    """

    contact_object_list = ["banana", "banana_01", "banana_02", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/bananas_3_bin.usda", contact_object_list)
    terminations = LeftBananaInBinTerminations
    instruction = {
        "default": "Put the left banana in the bin",
        "vague": "Put the banana in the bin",
        "referential": "Put that banana in the bin",
        "specific": "Put the left banana in the bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "banana": [
                    (partial(object_grabbed, object="banana"), 0.3),
                    (partial(object_picked_up, object="banana", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="banana",
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
            name="pick_and_place_banana",
        ),
    ]
