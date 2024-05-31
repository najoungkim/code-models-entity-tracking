from dataclasses import dataclass, field


@dataclass
class BasePrompt:
    """Base prompt class."""

    # Task instruction string
    instruction: str = "Given the description after \"Description:\", write a true statement about all boxes and their contents according to the description after \"Statement:\"."
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

    def prompt_prefix(self):
        """Prompt list with instruction and few-shot examples in chat format"""

        messages = []
        messages.append({"role": "system", "content": self.instruction})
        for input_str, output_str in self.few_shot_examples:
            messages.append(
                {"role": "user", "content": f"{self.input_prefix}{input_str}"})
            messages.append(
                {"role": "assistant", "content": f"{self.output_prefix} { output_str}"})

        return messages

    def get_prompt(self, test_input):
        """Prompt list formatted with the new input: test_input"""

        messages = self.prompt_prefix()
        messages.append(
            {"role": "user", "content": f"{self.input_prefix}{test_input}"})

        return messages


if __name__ == '__main__':
    print(BasePrompt().instruction)
