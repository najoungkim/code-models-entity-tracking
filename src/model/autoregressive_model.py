"""Implementation of conditional generation with Autoregressive Models implemented in Huugingface."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.base_model import BaseModel


class AutoregressiveModel(BaseModel):
    """Wrapper class for Autoregressive models in HF."""

    def __init__(self, model_str):
        super().__init__(model_str=model_str)

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_str)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)

    def generate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        )
        input_len = inputs[0].shape[1]
        outputs = self.model.generate(
            **inputs, max_length=self.model.config.max_position_embeddings)

        # Strip the prefix text
        text = self.tokenizer.batch_decode(outputs[input_len:])[0]
        output = text
        # output = text[len(prompt):].strip()
        return output


if __name__ == "__main__":
    from prompt.prompt_library import TwoShotPrompt
    output = AutoregressiveModel(
        "mistralai/Mistral-7B-Instruct-v0.1").generate(TwoShotPrompt.prompt_prefix())
    print(output)
