import argparse
import json
import torch
import os
import pandas as pd
from tqdm import tqdm
import transformers
from transformers.utils import logging

from model.autoregressive_model import AutoregressiveModel
from prompt.base_prompt import ChatPrompt
from prompt.prompt_library import TwoShotPrompt

logging.set_verbosity(transformers.logging.CRITICAL)

NUM_BOXES = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint", default=None, type=str, required=True,
                        help="Name of model to use (e.g., 't5-base') or a path that contains the model checkpoint.")
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to a directory that contains files of the form {split}-t5.jsonl")
    parser.add_argument("--output_path", default=None, type=str, required=True)

    parser.add_argument("--chat", action="store_true",
                        help="Set this flag for evaluating chat-based models")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    chat = args.chat
    if chat:
        prompt = ChatPrompt(few_shot_examples=TwoShotPrompt.few_shot_examples)
    else:
        prompt = TwoShotPrompt

    # Load datasets from path
    dataset_path = os.path.join(args.dataset_path, '{}-t5.jsonl')
    test_df = pd.read_json(dataset_path.format(
        'test-subsample-states'), orient='records', lines=True)

    # Set output path
    os.makedirs(args.output_path, exist_ok=True)
    predictions_path = os.path.join(args.output_path, "predictions.jsonl")

    model = AutoregressiveModel(args.model_name_or_checkpoint, device=device)

    with open(predictions_path, "w", encoding="UTF-8") as out_f:
        for idx, ex in tqdm(test_df.iterrows(), total=len(test_df)):
            # test in condensed format, so only consider every BOX_NUMBERth entry
            if idx % NUM_BOXES == 0:
                prefix = ex["sentence_masked"].split(" <extra_id_0>")[0][:-15]
                target = ex["masked_content"].replace("<extra_id_0> ", "")
                if chat:
                    messages = prompt.get_prompt(prefix)
                    # print(messages)
                    # break
                    response = model.chat_generate(messages)
                else:
                    input = prompt.get_prompt(prefix)
                    response = model.generate(input)

                # TODO: target is wrong since it only outputs one box
                # this is currently taken care of by expand_results.py
                # but would be cleaner to fix this here
                out = {"input": prefix, "target": target,
                       "prediction": response}
                print(json.dumps(out), file=out_f)


if __name__ == "__main__":
    main()
