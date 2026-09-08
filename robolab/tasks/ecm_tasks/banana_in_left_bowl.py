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
class BananaInLeftBowlTerminations:
    """Success when the banana is resting inside the left-hand red bowl.

    Dropping it into the right-hand bowl aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl_right",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl_left",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaInLeftBowlTask(Task):
    """Clarification target: one banana, two identical red bowls; the ground truth is the bowl on the robot's left.

    Under the ambiguous prompt pi05 uses the right-hand bowl by default (measured), so the
    baseline fails unless the destination is clarified.
    """

    contact_object_list = ["banana", "bowl_left", "bowl_right", "table"]
    scene = import_scene("ecm_scenes/banana_two_bowls.usda", contact_object_list)
    terminations = BananaInLeftBowlTerminations
    instruction = {
        "default": "Put the banana in the left bowl",
        "vague": "Put the banana in the bowl",
        "referential": "Put the banana in that bowl",
        "specific": "Put the banana in the left bowl",
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
                            container="bowl_left",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_banana_bowl_left",
        ),
    ]
