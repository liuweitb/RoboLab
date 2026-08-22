# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class RubiksCubeInBowlTerminations:
    """Success when the rubiks cube is resting inside the bowl and the gripper has released it.

    The banana is a distractor; placing it in the bowl does not succeed.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
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
class RubiksCubeInBowlTask(Task):
    contact_object_list = ["rubiks_cube", "banana", "bowl", "table"]
    # Explicit subfolder: a same-named scene also sits directly in assets/scenes/,
    # so a bare filename would resolve ambiguously.
    scene = import_scene("ecm_scenes/rubiks_cube_banana.usda", contact_object_list)
    terminations = RubiksCubeInBowlTerminations
    instruction = {
        "default": "Put the banana into the bowl",
        "vague": "Put it into the bowl",
        "specific": "Put the banana at the bottom of the bowl",
    }
    episode_length_s: int = 50
    attributes = ["semantics", "spatial"]

    # subtasks = [
    #     pick_and_place(object=["rubiks_cube"], container="bowl", logical="all", score=1.0)
    # ]
