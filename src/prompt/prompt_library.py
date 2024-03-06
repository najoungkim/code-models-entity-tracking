"""Instantiation of different prompts used in the experiments."""

# from prompt.base_prompt import BasePrompt

examples = [
    {"setting": "Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.",
     "state": "Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map."
     },
    {"setting": "Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.",
     "state": "Box 0 contains the plane, Box 1 contains the cross, Box 2 contains the bag and the machine and the map, Box 3 contains the coat, Box 4 contains nothing, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle."
     }
]


# ZeroShotPrompt = BasePrompt(
#     few_shot_examples=[],
# )

# TwoShotPrompt = BasePrompt(
#     few_shot_examples=orig_input_output_pairs
# )

# if __name__ == '__main__':
#     print(TwoShotPrompt.prompt_prefix())
