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
class BananaInRedBowlTerminations:
    """Success when the banana is resting inside the red bowl.

    Dropping it into the grey bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
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
            "object": "banana",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaInRedBowlTask(Task):
    """Clarification target: one banana, two containers (red bowl left, grey bin right); the ground truth is the bowl.

    Under the ambiguous prompt pi05 uses the grey bin on the right by default (measured), so the
    baseline fails unless the destination is clarified.
    """

    contact_object_list = ["banana", "bowl", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/banana_bowl_bin.usda", contact_object_list)
    terminations = BananaInRedBowlTerminations
    instruction = {
        "default": "Put the banana in the red bowl",
        "vague": "Put the banana in the container",
        "referential": "Put the banana in there",
        "specific": "Put the banana in the red bowl",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "color"]

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
            name="pick_and_place_banana_bowl",
        ),
    ]
