"""Prompt"""

from prompt.prompt_demonstrations import BASE_EXAMPLES
from prompt.prompt_structure import BasePrompt, ChatPrompt


def construct_prompt(prompt_structure="base", model_name=None):
    """Construct prompt with the structure and examples provided."""
    if prompt_structure == "base":
        return BasePrompt(examples=BASE_EXAMPLES)
    elif prompt_structure == "chat":
        return ChatPrompt(examples=BASE_EXAMPLES, model_name=model_name)
    else:
        raise (NotImplementedError("Current prompt not supported"))
