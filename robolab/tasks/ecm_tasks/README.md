# Available Tasks

This table contains metadata for all tasks in `robolab/tasks/ecm_tasks/`.

**Total Tasks: 7**

| task_name | scene | instruction | episode_s | attributes | num_subtasks | difficulty_label |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| BagelsInBowlBananasInPlateTask (bagels_in_bowl_bananas_in_plate.py) | No image | **default:** Pick up the yellow banana and place it flat on the white ceramic plate, then put the two bagels in the bowl<br>**referential:** sort things out<br>**intent:** help me prepare for the breakfast | 180 | semantics, sorting, spatial | 3 | complex |
| BananaInBowlTask (banana_in_bowl.py) | No image | **default:** Put the banana into the bowl<br>**vague:** Put it into the bowl<br>**specific:** Put the banana at the bottom of the bowl | 50 | semantics, spatial | 1 | simple |
| OneBananaInBowlTask (one_banana_in_bowl.py) | No image | **default:** Pick up the banana closest to the bowl and place it in the bowl<br>**vague:** Put the nearest piece of fruit in the bowl<br>**specific:** Three yellow bananas lie on the table. Grasp the one nearest the bowl, on the opposite side of the bowl from the other two, place it inside the bowl and release it | 300 | semantics, spatial | 1 | simple |
| PlaceWhiteFaceUpRubiksCubeTask (place_white_face_up_rubiks_cube.py) | No image | **default:** Pick up the Rubiks Cube with the white face on top and place it in the bowl<br>**referential:** Put that cube into the bowl<br>**spelling:** Puck that Magic Cube withe the whote surface on top and place it in the bowl | 60 | color, semantics, spatial | 1 | simple |
| PutMustardRightBinTask (put_mustard_right_bin.py) | No image | **default:** Pick up the mustard and put it in the right bin.<br>**referential:** Pick up the yellow stuff and put it in that bin<br>**intent:** I need to use the mustard later. | 60 | semantics, spatial | 1 | simple |
| PutPumpkinInBinTask (put_pumpkin_in_bin.py) | No image | **default:** Pick up the big and small pumpkins from the table and place them all into the bin<br>**referential:** Put two fruits and place them all into the bin<br>**spelling:** Puck up the pumppins from the table and place them all into the bun. | 120 | semantics, size, counting, spatial | 2 | moderate |
| RubiksCubeInBowlTask (rubiks_cube_in_bowl.py) | No image | **default:** Put the rubiks cube into the bowl<br>**vague:** Put it into the bowl<br>**specific:** Put the rubiks at the bottom of the bowl | 50 | semantics, spatial | 1 | simple |

This table was generated automatically from CSV data. Last updated: 2026-08-23 19:30:03