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
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=model_str,
        #     device=device,
        # )

    def generate(self, prompt):
        # add_special_tokens = False
        # if "gemma" in self.model_str:
        # add_special_tokens = True

        # outputs = self.pipeline(
        #     prompt,
        #     max_new_tokens=256,
        #     add_special_tokens=add_special_tokens,
        #     do_sample=False,
        #     temperature=None,
        #     top_p=None
        # )
        # generated_text = outputs[0]["generated_text"]
        # output_text = generated_text[len(prompt):]
        # return output_text

        cur_input_ids = self.tokenizer(prompt)["input_ids"]
        num_prompt_tokens = len(cur_input_ids)
        # cur_input_ids = list(input_ids)

        for box_id in range(NUM_BOXES):
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
            if eos_token in output_str:
                output_str = output_str[:output_str.index(eos_token) + 1]
            else:
                output_str = output_str + eos_token

            cur_input_ids += self.tokenizer(output_str,
                                            add_special_tokens=False)["input_ids"]

            # print(output_str)

        output_str = "Statement:" + self.tokenizer.decode(
            cur_input_ids[num_prompt_tokens:], skip_special_tokens=True)
        # print(output_str)
        return output_str

    def chat_generate(self, messages):
        """Generate response for chat-optimized model"""
        # if "gemma" in self.model_str:
        #     inputs = self.tokenizer.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True)
        # else:
        #
        # add_special_tokens = False
        # if "gemma" in self.model_str:
        add_special_tokens = True
        chat_fmt_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            add_special_tokens=add_special_tokens
        )

        return self.generate(chat_fmt_prompt)
        # input_len = inputs[0].shape[0]

        # outputs = self.pipeline(inputs, max_new_tokens=256, add_special_tokens=True, do_sample=False)

        # outputs =

        # return output


if __name__ == "__main__":
    from prompt.prompt_library import TwoShotPrompt
    output = AutoregressiveModel(
        "mistralai/Mistral-7B-Instruct-v0.1").generate(TwoShotPrompt.prompt_prefix())
    print(output)
