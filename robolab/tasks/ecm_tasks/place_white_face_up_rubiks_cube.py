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
class PlaceWhiteFaceUpRubiksCubeTerminations:
    """Success when the white-face-up cube (`rubiks_cube_target`) is resting inside the bowl.

    The other two cubes are distractors that differ only in which face points up:
    dropping either of them into the bowl aborts the episode as a truncation, so a
    wrong-cube run can never be scored as success.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["rubiks_cube", "rubiks_cube_01"],
            "container": "bowl",
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
            "object": "rubiks_cube_target",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PlaceWhiteFaceUpRubiksCubeTask(Task):
    """Task: place the Rubik's cube whose white face points up into the bowl."""

    contact_object_list = [
        "rubiks_cube_target",
        "rubiks_cube",
        "rubiks_cube_01",
        "bowl",
        "table",
    ]
    scene = import_scene("ecm_scenes/three_rubiks_cube_bowl.usda", contact_object_list)
    terminations = PlaceWhiteFaceUpRubiksCubeTerminations
    instruction = {
        "default": "Pick up the Rubiks Cube with the white face facing up and place it in the bowl",
        "referential": "Put that cube into the bowl",
        "spelling": "Puck that Magic Cube withe the whote surface facing up and place it in the bowl",
        "specific": "Pick up the Rubiks Cube with the white face facing up and place it in the bowl",
    }
    episode_length_s: int = 60
    attributes = ["color", "semantics", "spatial"]

    # Only the target cube counts: the two distractor cubes are deliberately absent
    # from the ladder, so manipulating them never advances the score.
    subtasks = [
        Subtask(
            conditions={
                "rubiks_cube_target": [
                    (partial(object_grabbed, object="rubiks_cube_target"), 0.3),
                    (partial(object_picked_up, object="rubiks_cube_target", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="rubiks_cube_target",
                            container="bowl",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_white_face_up_cube",
        ),
    ]
