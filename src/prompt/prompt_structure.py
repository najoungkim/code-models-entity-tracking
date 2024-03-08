"""Prompt structure."""

import outlines
from transformers import AutoTokenizer


@outlines.prompt
def base_few_shots(instruction, examples, new_query, answer_var="Statement"):
    """{{ instruction }}

    {% for example in examples %}
    Description: {{ example.input }}
    {{ answer_var }}: {{ example.output }}

    {% endfor %}
    Description: {{ new_query }}
    """


class BasePrompt:
    """Base prompt class."""
    instruction = 'Given the description after "Description:", write a true statement about all boxes and their contents to the description after "Statement:".'

    def __init__(self, examples):
        self.examples = examples
        self.show_prompt = True

    def get_few_shot_prompt(self, new_query):
        "Construct the few-shot prompt with instruction, examples, and the new query"
        prompt_string = base_few_shots(
            self.instruction, self.examples, new_query)

        # Show the first prompt
        if self.show_prompt:
            print(prompt_string)
            self.show_prompt = False

        return prompt_string


class ChatPrompt(BasePrompt):
    """Chat prompt class."""

    def __init__(self, examples, model_name):
        super(ChatPrompt, self).__init__(examples)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_few_shot_prompt(self, new_query):
        "Construct the few-shot prompt with instruction, examples, and the new query"
        messages = []

        # Add instruction
        if "Llama" in self.model_name:
            messages.append({"role": "system", "content": self.instruction})
        else:
            # Models other than Llama don't have system role
            messages.append({"role": "user", "content": self.instruction})
            messages.append({"role": "assistant", "content": "Okay."})

        for example in self.examples:
            messages.append(
                {"role": "user", "content": f"Description: {example['input']}"}
            )
            messages.append(
                {"role": "assistant",
                    "content": f"Statement: {example['output']}"}
            )

        messages.append(
            {"role": "user", "content": f"Description: {new_query}"})

        prompt_string = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_special_tokens=False, add_generation_prompt=True)

        # Show the first prompt
        if self.show_prompt:
            print(prompt_string)
            self.show_prompt = False

        return prompt_string
