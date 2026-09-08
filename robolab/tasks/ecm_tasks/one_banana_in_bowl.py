# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class OneBananaInBowlTerminations:
    """Success when banana_02 -- the banana lying closest to the bowl -- is in the bowl.

    The other two bananas are distractors; placing them in the bowl does not succeed.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "banana_02", "container": "bowl", "tolerance": 0.0, "require_contact_with": True, "require_gripper_detached": True}
    )

@dataclass
class OneBananaInBowlTask(Task):
    contact_object_list = ["banana", "banana_01", "banana_02", "bowl", "table"]
    scene = import_scene("banana_bowl_3.usda", contact_object_list)
    terminations = OneBananaInBowlTerminations
    instruction = {
        "default": "Pick up the banana closest to the bowl and place it in the bowl",
        "vague": "Put the nearest piece of fruit in the bowl",
        "referential": "Put that banana in the bowl",
        "specific": "Three yellow bananas lie on the table. Grasp the one nearest the bowl, on the opposite side of the bowl from the other two, place it inside the bowl and release it",
    }
    episode_length_s: int = 300
    attributes = ['semantics', 'spatial']

    subtasks = [
        pick_and_place(
            object=["banana_02"],
            container="bowl",
            logical="all",
            score=1.0
        )
    ]
