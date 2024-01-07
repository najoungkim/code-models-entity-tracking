"""Instantiation of different prompts used in the experiments."""

from prompt.base_prompt import BasePrompt

orig_input_output_pairs = [
    ("Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.",
     "Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map."
     ),
    ("Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.",
     "Box 0 contains the plane, Box 1 contains the cross, Box 2 contains the bag and the machine and the map, Box 3 contains the coat, Box 4 contains nothing, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle."
     )
]

complex_input_output_pairs = [
    ("Box 0 contains the green shoe and the yellow rose, Box 1 contains the blue bone and the big rose, Box 2 contains the red file and the yellow bell, Box 3 contains the red rose and the green tie, Box 4 contains the red bell and the big bell and the small letter, Box 5 contains the green rose and the yellow paper, Box 6 contains the small bell. Move the bell from Box 6 to Box 5. Move the bell from Box 2 to Box 6. Move the rose from Box 1 to Box 0. Move the file from Box 2 to Box 6. Remove the paper from Box 5. Remove the rose and the bell from Box 5. Move the red bell and the big bell from Box 4 to Box 1. Move the letter from Box 4 to Box 6.",
     "Box 0 contains the green shoe and the yellow rose and the big rose. Box 1 contains the blue bone and the red bell and the big bell. Box 2 contains nothing. Box 3 contains the red rose and the green tie. Box 4 contains nothing. Box 5 contains nothing. Box 6 contains the small letter and the red file and the yellow bell."),
    ("Box 0 contains the big cheese and the blue clock and the yellow engine, Box 1 contains the green bomb, Box 2 contains the big wire and the blue block and the blue bomb, Box 3 contains the big block, Box 4 contains the green apple and the green wheel, Box 5 contains the green clock and the red apple, Box 6 contains the blue wire and the green bone and the red engine. Remove the wire and the bomb from Box 2. Remove the cheese and the clock from Box 0. Move the engine from Box 0 to Box 1. Move the engine from Box 1 to Box 4.",
     "Box 0 contains nothing. Box 1 contains the green bomb. Box 2 contains the blue block. Box 3 contains the big block. Box 4 contains the green apple and the green wheel and the yellow engine. Box 5 contains the green clock and the red apple. Box 6 contains the blue wire and the green bone and the red engine.")
]


ZeroShotPrompt = BasePrompt(
    few_shot_examples=[],
)

TwoShotPrompt = BasePrompt(
    few_shot_examples=orig_input_output_pairs
)

ComplexTwoShotPrompt = BasePrompt(
    few_shot_examples=complex_input_output_pairs
)

FourShotPrompt = BasePrompt(
    few_shot_examples=orig_input_output_pairs + complex_input_output_pairs
)


if __name__ == '__main__':
    print(TwoShotPrompt.prompt_prefix())
