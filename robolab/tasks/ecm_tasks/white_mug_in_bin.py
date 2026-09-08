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
class WhiteMugInBinTerminations:
    """Success when the white mug (right of the row) is resting inside the grey bin.

    Dropping either other mug into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["red_mug", "ceramic_mug"],
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
            "object": "white_mug",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class WhiteMugInBinTask(Task):
    """Clarification target: three mugs of different colours; the ground truth is the white one.

    Under the ambiguous prompt pi05 grabs the centre blue-and-white ceramic mug by default (measured), so the
    baseline fails unless the ambiguity is resolved; the one-line specific prompt names the
    target by colour.
    """

    contact_object_list = ["red_mug", "ceramic_mug", "white_mug", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/mugs_three_bin.usda", contact_object_list)
    terminations = WhiteMugInBinTerminations
    instruction = {
        "default": "Put the white mug in the bin",
        "vague": "Put the mug in the bin",
        "referential": "Put that mug in the bin",
        "specific": "Put the white mug in the bin",
    }
    episode_length_s: int = 60
    attributes = ["color", "semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "white_mug": [
                    (partial(object_grabbed, object="white_mug"), 0.3),
                    (partial(object_picked_up, object="white_mug", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="white_mug",
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
            name="pick_and_place_white_mug",
        ),
    ]
