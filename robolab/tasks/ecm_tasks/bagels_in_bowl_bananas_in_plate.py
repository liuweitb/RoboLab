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
    object_on_top,
    object_picked_up,
)
from robolab.core.task.decorators import atomic
from robolab.core.task.predicate_logic import _and
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

BAGELS = ["bagel_00", "bagel_06"]


@atomic
def bagels_in_bowl_and_banana_on_plate(env, env_id: int | None = None):
    """Both bagels sit in the bowl AND the banana rests flat on the large plate.

    The two halves of the task target different geometry — an open-top container versus a
    flat surface — so neither `object_in_container` nor `object_on_top` can express the
    goal alone. Conjoining them here keeps success a single termination term.
    """
    bagels_placed = object_in_container(
        env,
        object=BAGELS,
        container="bowl",
        tolerance=0.0,
        require_contact_with=False,
        require_gripper_detached=True,
        logical="all",
        env_id=env_id,
    )
    banana_placed = object_on_top(
        env,
        object="banana",
        reference_object="plate_large",
        require_gripper_detached=True,
        env_id=env_id,
    )
    return _and(bagels_placed, banana_placed)


@configclass
class BagelsInBowlBananasInPlateTerminations:
    """Success when both bagels are in the bowl and the banana is flat on the large plate.

    The two destinations are swappable by mistake, so each swap is an explicit truncation:
    a bagel landing on the plate or the banana landing in the bowl aborts the episode and
    can never be scored as success.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    undesired_bagel_on_plate = DoneTerm(
        func=object_on_top,
        params={
            "object": BAGELS,
            "reference_object": "plate_large",
            "require_gripper_detached": True,
            "logical": "any",
        },
        time_out=True,
    )
    undesired_banana_in_bowl = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
        },
        time_out=True,
    )
    success = DoneTerm(func=bagels_in_bowl_and_banana_on_plate, params={})


@dataclass
class BagelsInBowlBananasInPlateTask(Task):
    """Task: bagels go in the bowl, the banana goes on the large plate."""

    contact_object_list = ["banana", "bagel_00", "bagel_06", "bowl", "plate_large", "table"]
    scene = import_scene("ecm_scenes/bagel_plate_banana_bowl.usda", contact_object_list)
    terminations = BagelsInBowlBananasInPlateTerminations
    instruction = {
        "default": (
            "Pick up the yellow banana and place it flat on the white ceramic plate, "
            "then put the two bagels in the bowl"
        ),
        "referential": "sort things out",
        "intent": "help me prepare for the breakfast",
        "specific": (
            "Pick up the yellow banana and place it flat on the white ceramic plate, "
            "then put the two bagels in the bowl"
        ),
    }
    episode_length_s: int = 180
    attributes = ["semantics", "sorting", "spatial"]

    # Half credit for the pair of bagels, half for the banana. `require_contact_with` is
    # left False for the bowl because the second bagel commonly lands on top of the first
    # and so never touches the bowl itself.
    subtasks = [
        Subtask(
            conditions={
                bagel: [
                    (partial(object_grabbed, object=bagel), 0.3),
                    (partial(object_picked_up, object=bagel, surface="table"), 0.3),
                    (
                        partial(
                            object_in_container,
                            object=bagel,
                            container="bowl",
                            tolerance=0.0,
                            require_contact_with=False,
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ]
                for bagel in BAGELS
            },
            logical="all",
            score=0.5,
            name="place_bagels_in_bowl",
        ),
        Subtask(
            conditions={
                "banana": [
                    (partial(object_grabbed, object="banana"), 0.3),
                    (partial(object_picked_up, object="banana", surface="table"), 0.3),
                    (
                        partial(
                            object_on_top,
                            object="banana",
                            reference_object="plate_large",
                            require_gripper_detached=True,
                        ),
                        0.4,
                    ),
                ],
            },
            logical="all",
            score=0.5,
            name="place_banana_on_plate",
        ),
    ]
