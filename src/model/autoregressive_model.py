"""Implementation of conditional generation with Autoregressive Models implemented in Huugingface."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.base_model import BaseModel


class AutoregressiveModel(BaseModel):
    """Wrapper class for Autoregressive models in HF."""

    def __init__(self, model_str, device="cpu"):
        super().__init__(model_str=model_str, device=device)

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_str).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str)

    def generate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False).to(self.device)
        input_len = inputs[0].shape[1]
        outputs = self.model.generate(
            **inputs, max_length=self.model.config.max_position_embeddings)

        # Strip the prefix text
        text = self.tokenizer.batch_decode(outputs[input_len:])[0]
        output = text
        # output = text[len(prompt):].strip()
        return output

    def chat_generate(self, messages):
        """Generate response for chat-optimized model"""

        print(messages)
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
