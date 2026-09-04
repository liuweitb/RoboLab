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
    object_on_top,
    object_picked_up,
)
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task


@configclass
class CeramicMugCupOnPlateTerminations:
    """Success when the blue ceramic mug is resting on the clay plate.

    The red mug and the white ribbed mug are same-category distractors: setting either
    on the plate aborts the episode as a truncation.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_behavior = DoneTerm(
        func=object_on_top,
        params={
            "object": ["red_mug", "mug"],
            "reference_object": "clay_plates",
            "require_gripper_detached": True,
            "logical": "any",
        },
        time_out=True,
    )
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": "ceramic_mug",
            "reference_object": "clay_plates",
            "require_gripper_detached": True,
        },
    )


@dataclass
class CeramicMugCupOnPlateTask(Task):
    """Clarification target: three drinking vessels that all count as "the cup".

    "Put the cup on the plate" fits all three mugs. The ground truth is the
    blue-and-white ceramic mug in the middle.
    """

    contact_object_list = ["red_mug", "ceramic_mug", "mug", "clay_plates", "table"]
    scene = import_scene("ecm_scenes/cups_clay_plate.usda", contact_object_list)
    terminations = CeramicMugCupOnPlateTerminations
    instruction = {
        "default": "Put the blue ceramic mug on the clay plate",
        "vague": "Put the cup on the plate",
        "referential": "Put that cup on the plate",
        "specific": "A red mug, a blue-and-white ceramic mug and a white ribbed mug sit in a row in front of a clay plate. Pick up the blue-and-white mug in the middle and set it down upright on the plate",
    }
    episode_length_s: int = 60
    attributes = ["semantics"]

    # Only the target advances the score; distractors are deliberately absent from the ladder.
    subtasks = [
        Subtask(
            conditions={
                "ceramic_mug": [
                    (partial(object_grabbed, object="ceramic_mug"), 0.3),
                    (partial(object_picked_up, object="ceramic_mug", surface="table"), 0.3),
                    (
                        partial(
                            object_on_top,
                            object="ceramic_mug",
                            reference_object="clay_plates",
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            name="pick_and_place_ceramic_mug_on_plate",
        ),
    ]
