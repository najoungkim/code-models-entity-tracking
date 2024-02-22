from dataclasses import dataclass, field


@dataclass
class BasePrompt:
    """Base prompt class."""

    # Task instruction string
    instruction: str = "Given the description after \"Description:\", write a true statement about all boxes and their contents to the description after \"Statement:\"."
    # List of few-shot input-output string pairs
    few_shot_examples: list = field(default_factory=list)

    # String separators
    # String separating instruction from I/O pairs
    instruction_separator: str = "\n\n"
    # String separating input-output pairs
    io_separator: str = "\n\n"
    # String separating examples
    example_separator: str = "\n\n"

    # Prefix for test input
    input_prefix: str = "Description: "
    # Prefix for test output
    # no trailing space here since this can potentially mess up tokenization
    output_prefix: str = "Statement:"

    def prompt_prefix(self):
        """Prompt prefix with instruction and few-shot examples"""
        prefix = self.instruction + self.instruction_separator
        for input_str, output_str in self.few_shot_examples:
            prefix += (
                self.input_prefix +
                input_str +
                self.io_separator +
                self.output_prefix +
                " " +
                output_str +
                self.example_separator
            )
        return prefix

    def get_prompt(self, test_input):
        """Return prompt formatted with the new input: test_input"""
        prompt_prefix = self.prompt_prefix()
        # prompt = ""
        prompt = (
            prompt_prefix +
            self.input_prefix +
            test_input +
            self.io_separator +
            self.output_prefix
        )
        return prompt


class ChatPrompt(BasePrompt):
    """Prompt class for chat-optimized models."""

    def __init__(self, few_shot_examples=None, model_str=None):
        super().__init__()
        self.few_shot_examples = few_shot_examples
        self.model_str = model_str

    def prompt_prefix(self):
        """Prompt list with instruction and few-shot examples in chat format"""

        messages = []
        if "mistralai/Mistral-7B-Instruct" in self.model_str:
            # (Shubham) Apparently there's no system role in Mistral models!
            # Was getting this error: Conversation roles must alternate user/assistant/user/assistant/
            # On a small subset, providing instruction as a chat between user and assistant did better
            # than not providing the instruction at all.
            messages.append({"role": "user", "content": self.instruction})
            messages.append({"role": "assistant", "content": "Okay."})
        elif "gemma" in self.model_str:
            messages.append({"role": "user", "content": self.instruction})
        else:
            messages.append({"role": "system", "content": self.instruction})

        for input_str, output_str in self.few_shot_examples:
            messages.append(
                {"role": "user", "content": f"{self.input_prefix}{input_str}"})
            messages.append(
                {"role": "assistant", "content": f"{self.output_prefix} {output_str}"})

        return messages

    def get_prompt(self, test_input):
        """Prompt list formatted with the new input: test_input"""

        messages = self.prompt_prefix()
        messages.append(
            {"role": "user", "content": f"{self.input_prefix}{test_input}"})

        return messages


if __name__ == '__main__':
    print(ChatPrompt("").prompt_prefix())
