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
class PickUpThatLemonTerminations:
    """Success when the lemon is held and lifted at least 10 cm above the table.

    No truncation is defined for lifting the cube or the can: a clarifying gesture may
    briefly lift a candidate, and that must not end the episode. The 10 cm threshold keeps
    a brief indication lift from counting as completion.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_picked_up,
        params={
            "object": "lemon_01",
            "surface": "table",
            "distance": 0.1,
        },
    )


@dataclass
class PickUpThatLemonTask(Task):
    """Clarification target: a purely deictic request with three unrelated candidates.

    "Pick that up" identifies nothing on its own. The ground truth is the lemon on the
    robot's left; the clarifier should ask "this one?" while pointing or touching.
    """

    contact_object_list = ["lemon_01", "rubiks_cube", "soup_can", "table"]
    scene = import_scene("ecm_scenes/pick_that_one.usda", contact_object_list)
    terminations = PickUpThatLemonTerminations
    instruction = {
        "default": "Pick up the lemon and hold it up",
        "vague": "Pick that up",
        "referential": "Grab that one",
        "specific": "A lemon, a Rubik's cube and a soup can sit on the table. Grasp the yellow lemon on the left and lift it well clear of the table",
    }
    episode_length_s: int = 45
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "lemon_01": [
                    (partial(object_grabbed, object="lemon_01"), 0.5),
                    (
                        partial(
                            object_picked_up,
                            object="lemon_01",
                            surface="table",
                            distance=0.1,
                        ),
                        0.5,
                    ),
                ],
            },
            logical="all",
            name="pick_up_lemon",
        ),
    ]
