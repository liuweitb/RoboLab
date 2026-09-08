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
class RubiksCubeInRedBowlTerminations:
    """Success when the Rubik's cube is resting inside the red bowl.

    Dropping it into the grey bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "rubiks_cube",
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
            "object": "rubiks_cube",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class RubiksCubeInRedBowlTask(Task):
    """Clarification target: one cube, two containers (red bowl left, grey bin right); the ground truth is the bowl.

    Under the ambiguous prompt pi05 uses the grey bin on the right by default (measured), so the
    baseline fails unless the destination is clarified.
    """

    contact_object_list = ["rubiks_cube", "bowl", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/rubiks_cube_bowl_bin.usda", contact_object_list)
    terminations = RubiksCubeInRedBowlTerminations
    instruction = {
        "default": "Put the cube in the red bowl",
        "vague": "Put the cube in the container",
        "referential": "Put the cube in there",
        "specific": "Put the cube in the red bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "color"]

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
            name="pick_and_place_rubiks_cube_bowl",
        ),
    ]
