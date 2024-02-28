"""Experiments for autoregressive model."""

import argparse
import json
import os
import torch
import pandas as pd
from tqdm import tqdm
import transformers
from transformers.utils import logging
from transformers import pipeline

from model.autoregressive_model import AutoregressiveModel
from prompt.base_prompt import ChatPrompt
from prompt.prompt_library import TwoShotPrompt, FourShotPrompt

import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message='TypedStorage is deprecated')

logging.set_verbosity(transformers.logging.CRITICAL)

NUM_BOXES = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint", default=None, type=str, required=True,
                        help="Name of model to use (e.g., 't5-base') or a path that contains the model checkpoint.")
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file.")
    parser.add_argument("--output_path", default=None, type=str, required=True)
    parser.add_argument("--k_shot", default=2, type=int, choices=[2, 4],
                        help="Number of examples in prompt.")
    parser.add_argument("--chat", action="store_true",
                        help="Set this flag for evaluating chat-based models.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_class = TwoShotPrompt if args.k_shot == 2 else FourShotPrompt
    chat = args.chat
    if chat:
        prompt = ChatPrompt(few_shot_examples=prompt_class.few_shot_examples,
                            model_str=args.model_name_or_checkpoint)
    else:
        prompt = prompt_class

    # Load datasets from path
    test_df = pd.read_json(args.test_file, orient='records', lines=True)

    # Set output path
    os.makedirs(args.output_path, exist_ok=True)
    predictions_path = os.path.join(args.output_path, "predictions.jsonl")

    model = AutoregressiveModel(args.model_name_or_checkpoint, device=device)

    with open(predictions_path, "w", encoding="UTF-8") as out_f:
        for idx, ex in tqdm(test_df.iterrows(), total=len(test_df)):
            # test in condensed format, so only consider every BOX_NUMBERth entry
            if idx % NUM_BOXES == 0:
                prefix = ex["sentence_masked"].split(" <extra_id_0>")[0][:-15]
                if chat:
                    query = prompt.get_prompt(prefix)
                    response = model.chat_generate(query)
                else:
                    query = prompt.get_prompt(prefix)
                    response = model.generate(query)

                out = {"input": prefix, "prediction": response}
                print(json.dumps(out), file=out_f)


if __name__ == "__main__":
    main()
