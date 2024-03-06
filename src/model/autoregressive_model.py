"""Implementation of conditional generation with Autoregressive Models implemented in Huugingface."""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from model.base_model import BaseModel
import torch


NUM_BOXES = 7


class AutoregressiveModel(BaseModel):
    """Wrapper class for Autoregressive models in HF."""

    def __init__(self, model_str, device="cpu"):
        super().__init__(model_str=model_str, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_str, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_str).to(device=device)

    def generate(self, prompt):
        cur_input_ids = self.tokenizer(prompt)["input_ids"]
        num_prompt_tokens = len(cur_input_ids)

        for box_id in range(NUM_BOXES):
            if box_id == 0:
                phrase = "Statement: Box 0 contains"
            else:
                phrase = f" Box {box_id} contains"
            tokenized_phrase = self.tokenizer(
                phrase, add_special_tokens=False)["input_ids"]

            model_output = self.model.generate(
                torch.tensor([cur_input_ids + tokenized_phrase]
                             ).to(self.device),
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

            num_prefix_toks = len(cur_input_ids)
            output_str = self.tokenizer.decode(
                model_output[0][num_prefix_toks:])
            eos_token = "," if box_id < 6 else "."

            # Trim the string to tokens like ",", "."
            eos_cands = []
            for eos_string in [",", "."]:
                if eos_string in output_str:
                    eos_cands.append(output_str.index(eos_string))

            if len(eos_cands) > 0:
                min_eos_idx = min(eos_cands)
                output_str = output_str[:min_eos_idx]

            output_str = output_str + eos_token
            cur_input_ids += self.tokenizer(output_str,
                                            add_special_tokens=False)["input_ids"]

        output_str = self.tokenizer.decode(
            cur_input_ids[num_prompt_tokens:], skip_special_tokens=True)
        # print(output_str)
        # print(self.tokenizer.decode(cur_input_ids, skip_special_tokens=False))
        return output_str


if __name__ == "__main__":
    from prompt.prompt_library import TwoShotPrompt
    output = AutoregressiveModel(
        "mistralai/Mistral-7B-Instruct-v0.1").generate(TwoShotPrompt.prompt_prefix())
    print(output)
