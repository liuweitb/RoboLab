# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import (
    object_above_bottom,
    object_dropped,
    object_grabbed,
    object_in_container,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class BananaInBowlTermination:
    """Success when the banana is resting inside the bowl and the gripper has released it.

    The rubiks cube is a distractor: placing it in the bowl aborts the episode as a
    truncation, so it can never be scored as success.
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
            "object": "banana",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaInBowlTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    # Explicit subfolder: a same-named scene also sits directly in assets/scenes/,
    # so a bare filename would resolve ambiguously.
    scene = import_scene("ecm_scenes/rubiks_cube_banana.usda", contact_object_list)
    terminations = BananaInBowlTermination
    instruction = {
        "default": "Put the banana into the bowl",
        "vague": "Put it into the bowl",
        "referential": "Put that into the bowl",
        "specific": "Put the banana at the bottom of the bowl",
    }
    episode_length_s: int = 50
    attributes = ["semantics", "spatial"]

    subtasks = [
        Subtask(
            conditions={
                "banana": [
                    (partial(object_grabbed, object="banana"), 0.1),
                    (partial(object_above_bottom, object="banana", reference_object="bowl"), 0.2),
                    (partial(object_dropped, object="banana"), 0.3),
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
            name="pick_and_place_banana",
        ),
    ]
