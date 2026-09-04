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
class KetchupSauceInBinTerminations:
    """Success when the ketchup bottle is resting inside the grey bin.

    The mustard and barbecue sauce bottles are same-shape distractors; dropping either
    into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["mustard_bottle", "bbq_sauce_bottle"],
            "container": "bin_b03",
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
            "object": "ketchup_bottle",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class KetchupSauceInBinTask(Task):
    """Clarification target: three squeeze bottles that are all "sauce".

    "Put the sauce in the bin" fits all three bottles; the intent variant is vaguer still.
    The ground truth is the red ketchup bottle on the robot's left.
    """

    contact_object_list = ["ketchup_bottle", "mustard_bottle", "bbq_sauce_bottle", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/sauces_bin.usda", contact_object_list)
    terminations = KetchupSauceInBinTerminations
    instruction = {
        "default": "Pick up the ketchup bottle and put it in the bin",
        "vague": "Put the sauce in the bin",
        "referential": "Put that bottle in the bin",
        "intent": "I need something for my fries, put it in the bin for me",
        "specific": "Three squeeze bottles sit in a row: red ketchup, yellow mustard and brown barbecue sauce. Pick up the red ketchup bottle on the left and drop it in the grey bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "color"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "ketchup_bottle": [
                    (partial(object_grabbed, object="ketchup_bottle"), 0.3),
                    (partial(object_picked_up, object="ketchup_bottle", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="ketchup_bottle",
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
            name="pick_and_place_ketchup",
        ),
    ]
