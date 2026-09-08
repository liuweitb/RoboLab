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
class PutMustardRightBinTerminations:
    """Success when the mustard bottle is resting inside the right-hand grey bin.

    The left-hand bin is the distractor target: dropping the mustard there aborts the
    episode as a truncation, so a left/right mix-up can never be scored as success.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_in_container,
        params={
            "object": "mustard",
            "container": "grey_bin_left",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "mustard",
            "container": "grey_bin_right",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PutMustardRightBinTask(Task):
    """Task: put the mustard bottle into the right-hand of two identical grey bins."""

    contact_object_list = ["mustard", "grey_bin_right", "grey_bin_left", "table"]
    scene = import_scene("ecm_scenes/two_bin_mustard.usda", contact_object_list)
    terminations = PutMustardRightBinTerminations
    instruction = {
        "default": "Pick up the mustard and put it in the right bin.",
        "referential": "Pick up the yellow stuff and put it in that bin",
        "intent": "I need to use the mustard later.",
        "specific": "Pick up the mustard and put it in the right bin.",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "spatial"]

    subtasks = [
        Subtask(
            conditions={
                "mustard": [
                    (partial(object_grabbed, object="mustard"), 0.3),
                    (partial(object_picked_up, object="mustard", surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object="mustard",
                            container="grey_bin_right",
                            tolerance=0.0,
                            require_contact_with=True,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_mustard_right_bin",
        ),
    ]
