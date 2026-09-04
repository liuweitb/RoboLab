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
class TomatoSoupCanInBinTerminations:
    """Success when the tomato soup can is resting inside the grey bin.

    The corn can and the green beans can are same-category distractors; dropping either
    into the bin aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": ["corn_can", "green_beans_can"],
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
            "object": "tomato_soup_can",
            "container": "bin_b03",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class TomatoSoupCanInBinTask(Task):
    """Clarification target: three cans with different labels.

    "Put the can in the bin" does not say which can. The ground truth is the tall red
    tomato soup can on the robot's left.
    """

    contact_object_list = ["tomato_soup_can", "corn_can", "green_beans_can", "bin_b03", "table"]
    scene = import_scene("ecm_scenes/cans_bin.usda", contact_object_list)
    terminations = TomatoSoupCanInBinTerminations
    instruction = {
        "default": "Pick up the tomato soup can and put it in the bin",
        "vague": "Put the can in the bin",
        "referential": "Put that can in the bin",
        "specific": "Three cans sit in a row: a tall red tomato soup can, a yellow corn can and a green beans can. Pick up the tall red soup can on the left and drop it in the grey bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "tomato_soup_can": [
                    (partial(object_grabbed, object="tomato_soup_can"), 0.3),
                    (partial(object_picked_up, object="tomato_soup_can", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="tomato_soup_can",
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
            name="pick_and_place_soup_can",
        ),
    ]
