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
class RubiksCubeInBinContainerTerminations:
    """Success when the Rubik's cube is resting inside the grey bin.

    The red bowl is the distractor destination: dropping the cube into it aborts the
    episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "rubiks_cube",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "rubiks_cube",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class RubiksCubeInBinContainerTask(Task):
    """Clarification target: the word "container" covers both a bowl and a bin.

    "Put the rubiks cube in the container" fits either receptacle. The ground truth is
    the grey metal bin on the robot's right.
    """

    contact_object_list = ["rubiks_cube", "bowl", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/rubiks_cube_bowl_bin.usda", contact_object_list)
    terminations = RubiksCubeInBinContainerTerminations
    instruction = {
        "default": "Put the rubiks cube in the grey bin",
        "vague": "Put the rubiks cube in the container",
        "referential": "Put the cube in there",
        "specific": "A red bowl and a grey metal bin sit behind a Rubik's cube. Pick up the cube and drop it in the grey metal bin on the right",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "rubiks_cube": [
                    (partial(object_grabbed, object="rubiks_cube"), 0.3),
                    (partial(object_picked_up, object="rubiks_cube", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="rubiks_cube",
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
            name="pick_and_place_cube_in_bin",
        ),
    ]
