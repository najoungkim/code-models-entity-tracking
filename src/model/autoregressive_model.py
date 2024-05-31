"""Implementation of conditional generation with Autoregressive Models implemented in Huugingface."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.base_model import BaseModel
import hf_olmo
import torch

class AutoregressiveModel(BaseModel):
    """Wrapper class for Autoregressive models in HF."""

    def __init__(self, model_str, quantization_config, device="cpu", hf_key=None):
<<<<<<< HEAD
        super().__init__(model_str=model_str,
=======
        super().__init__(
            model_str=model_str,
>>>>>>> d7ed306aa798904525fe72b5dbf6d1344be02839
            quantization_config=quantization_config,
            device=device,
            hf_key=None)

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_str,
            quantization_config=self.quantization_config,
<<<<<<< HEAD
            use_auth_token=self.hf_key,
            torch_dtype=torch.float16 if self.quantization_config else None)
=======
            use_auth_token=self.hf_key)
>>>>>>> d7ed306aa798904525fe72b5dbf6d1344be02839
        if self.quantization_config is None:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_str, use_auth_token=self.hf_key)

    def generate(self, prompt):
        add_kwargs = {}
        if "olmo" in self.model_str.lower():
            add_kwargs["return_token_type_ids"] = False
        inputs = self.tokenizer(
            prompt, return_tensors="pt", **add_kwargs)
        input_len = len(inputs[0].ids)
        outputs = self.model.generate(
            **inputs.to(self.device), max_new_tokens=512) #, max_length=self.model.config.max_position_embeddings)

        # Strip the prefix text
        tokens = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True)
        text = "".join(tokens)

        return text

    def chat_generate(self, messages):
        """Generate response for chat-optimized model"""

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt").to(self.device)
        
        input_len = inputs[0].shape[0]

        outputs = self.model.generate(
            inputs, max_new_tokens=512)#, max_length=self.model.config.max_position_embeddings)

        output = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True)

        return output


if __name__ == "__main__":
    from prompt.prompt_library import TwoShotPrompt
    output = AutoregressiveModel(
        "mistralai/Mistral-7B-Instruct-v0.1").generate(TwoShotPrompt.prompt_prefix())
    print(output)
