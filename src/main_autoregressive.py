"""Experiments for autoregressive model."""

import argparse
import json
import os
import warnings
import torch
import pandas as pd
import outlines
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from transformers.utils import logging

from prompt import construct_prompt
from constants import REGEX_EXPR, NUM_BOXES

warnings.filterwarnings('ignore', category=UserWarning,
                        message='TypedStorage is deprecated')

logging.set_verbosity(transformers.logging.CRITICAL)


def main():
    """Parse arguments and start inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Name of model to use (e.g., 't5-base') or a path that contains the model checkpoint.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test file.")
    parser.add_argument("--output_path", default=None, type=str, required=True)
    parser.add_argument("--chat", action="store_true",
                        help="Set this flag for evaluating chat-based models.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model_kwargs = {"trust_remote_code": True}
    if transformers.utils.is_torch_bf16_gpu_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    print(args.model_name)
    model = outlines.models.transformers(
        args.model_name, device=device, model_kwargs=model_kwargs)

    # Initialize prompt
    if args.chat:
        prompt = construct_prompt(
            prompt_structure="chat", model_name=args.model_name)
    else:
        prompt = construct_prompt()

    # Load datasets from path
    test_df = pd.read_json(args.test_file, orient='records', lines=True)

    # Set output path
    os.makedirs(args.output_path, exist_ok=True)
    predictions_path = os.path.join(args.output_path, "predictions.jsonl")

    generator = outlines.generate.regex(
        model, REGEX_EXPR, sampler=outlines.samplers.GreedySampler())

    with open(predictions_path, "w", encoding="UTF-8") as out_f:
        for idx, ex in tqdm(test_df.iterrows(), total=len(test_df)):
            # test in condensed format, so only consider every BOX_NUMBERth entry
            if idx % NUM_BOXES == 0:
                prefix = ex["sentence_masked"].split(" <extra_id_0>")[0][:-15]
                query = prompt.get_few_shot_prompt(prefix)
                response = generator(query, max_tokens=100)

                out = {"input": prefix, "prediction": response}
                print(json.dumps(out), file=out_f)


if __name__ == "__main__":
    main()
