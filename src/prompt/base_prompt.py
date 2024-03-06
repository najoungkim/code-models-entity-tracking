import outlines


@outlines.prompt
def base_few_shots(instruction, examples, setting):
    """{{ instruction }}

    {% for example in examples %}
    Description: {{ example.setting }}
    Statement: {{ example.state }}

    {% endfor %}
    Description: {{ setting }}
    Statement: 
    """


class BasePrompt:
    instruction = 'Given the description after "Description:", write a true statement about all boxes and their contents to the description after "Statement:".'

    def __init__(self, examples):
        self.examples = examples

    def get_few_shot_prompt(self, setting):
        return base_few_shots(self.instruction, self.examples, setting)

# class ChatPrompt(BasePrompt):
#     """Prompt class for chat-optimized models."""

#     def __init__(self, few_shot_examples=None, model_str=None):
#         super().__init__()
#         self.few_shot_examples = few_shot_examples
#         self.model_str = model_str

#     def prompt_prefix(self):
#         """Prompt list with instruction and few-shot examples in chat format"""

#         messages = []
#         if "mistralai/Mistral-7B-Instruct" in self.model_str:
#             # (Shubham) Apparently there's no system role in Mistral models!
#             # Was getting this error: Conversation roles must alternate user/assistant/user/assistant/
#             # On a small subset, providing instruction as a chat between user and assistant did better
#             # than not providing the instruction at all.
#             messages.append({"role": "user", "content": self.instruction})
#             messages.append({"role": "assistant", "content": "Okay."})
#         elif "gemma" in self.model_str:
#             messages.append({"role": "user", "content": self.instruction})
#             messages.append({"role": "assistant", "content": ""})
#             # pass
#         else:
#             messages.append({"role": "system", "content": self.instruction})

#         for input_str, output_str in self.few_shot_examples:
#             messages.append(
#                 {"role": "user", "content": f"{self.input_prefix}{input_str}"})
#             messages.append(
#                 {"role": "assistant", "content": f"{self.output_prefix}{output_str}"})

#         return messages

#     def get_prompt(self, test_input):
#         """Prompt list formatted with the new input: test_input"""

#         messages = self.prompt_prefix()
#         messages.append(
#             {"role": "user", "content": f"{self.input_prefix}{test_input}"})

#         return messages


# if __name__ == '__main__':
#     print(ChatPrompt("").prompt_prefix())
