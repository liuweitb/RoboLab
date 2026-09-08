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
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class PickUpRubiksCubeTerminations:
    """Success when the Rubik's cube is held and lifted at least 10 cm above the table.

    No truncation for lifting the lemon or the can: a clarifying gesture may briefly lift a
    candidate. The 10 cm threshold keeps an indication lift from counting as completion.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "rubiks_cube",
            "surface": "table",
            "distance": 0.1,
        },
    )


@dataclass
class PickUpRubiksCubeTask(Task):
    """Clarification target: a purely deictic request with three unrelated candidates; the ground
    truth is the Rubik's cube.

    Under the ambiguous prompt pi05 grabs the lemon by default (measured), so the baseline fails
    unless the ambiguity is resolved; the one-line specific prompt names the cube.
    """

    contact_object_list = ["lemon_01", "rubiks_cube", "soup_can", "table"]
    scene = import_scene("ecm_scenes/pick_that_one.usda", contact_object_list)
    terminations = PickUpRubiksCubeTerminations
    instruction = {
        "default": "Pick up the Rubik's cube",
        "vague": "Pick that up",
        "referential": "Grab that one",
        "specific": "Pick up the Rubik's cube",
    }
    episode_length_s: int = 45
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "rubiks_cube": [
                    (partial(object_grabbed, object="rubiks_cube"), 0.5),
                    (
                        partial(
                            object_picked_up,
                            object="rubiks_cube",
                            surface="table",
                            distance=0.1,
                        ),
                        0.5,
                    ),
                ],
            },
            logical="all",
            name="pick_up_rubiks_cube",
        ),
    ]
