# Available Tasks

This table contains metadata for all tasks in `robolab/tasks/ecm_tasks`.

**Total Tasks: 2**

| task_name | scene | instruction | episode_s | attributes | num_subtasks | difficulty_label |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| OneBananaInBowlTask (one_banana_in_bowl.py) | No image | **default:** Pick up the banana closest to the bowl and place it in the bowl<br>**vague:** Put the nearest piece of fruit in the bowl<br>**specific:** Three yellow bananas lie on the table. Grasp the one nearest the bowl, on the opposite side of the bowl from the other two, place it inside the bowl and release it | 300 | semantics, spatial | 1 | simple |
| RubiksCubeInBowlTask (rubiks_cube_in_bowl.py) | No image | **default:** Put the banana into the bowl<br>**vague:** Put it into the bowl<br>**specific:** Put the banana at the bottom of the bowl | 50 | semantics, spatial | 1 | simple |

This table was generated automatically from CSV data. Last updated: 2026-08-21 17:54:18