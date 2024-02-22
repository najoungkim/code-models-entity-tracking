"""Implementation of conditional generation with Autoregressive Models implemented in Huugingface."""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from model.base_model import BaseModel


class AutoregressiveModel(BaseModel):
    """Wrapper class for Autoregressive models in HF."""

    def __init__(self, model_str, device="cpu"):
        super().__init__(model_str=model_str, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_str, trust_remote_code=True)
        self.pipeline = pipeline(
            "text-generation",
            model=model_str,
            device=device,
        )

    # def initialize_model(self):

    def generate(self, prompt):
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_len = len(inputs[0].ids)
        # outputs = self.model.generate(
        #     **inputs.to(self.device), max_new_tokens=512, max_length=self.model.config.max_position_embeddings)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            add_special_tokens=True,
            do_sample=False,
        )
        generated_text = outputs[0]["generated_text"]

        # Strip the prefix text
        # tokens = self.tokenizer.decode(
        #     generated_text[len(prompt):], skip_special_tokens=True)
        # text = "".join(tokens)
        output_text = generated_text[len(prompt):]
        return output_text

    def chat_generate(self, messages):
        """Generate response for chat-optimized model"""
        if "gemma" in self.model_str:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt").to(self.device)

        input_len = inputs[0].shape[0]

        outputs = self.model.generate(
            inputs, max_new_tokens=512, max_length=self.model.config.max_position_embeddings)

        output = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True)

        return output


if __name__ == "__main__":
    from prompt.prompt_library import TwoShotPrompt
    output = AutoregressiveModel(
        "mistralai/Mistral-7B-Instruct-v0.1").generate(TwoShotPrompt.prompt_prefix())
    print(output)
