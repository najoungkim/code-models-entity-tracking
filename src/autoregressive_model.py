import argparse
import json
import torch
import os
import pandas as pd

from model.autoregressive_model import AutoregressiveModel
from prompt.base_prompt import ChatPrompt
from prompt.prompt_library import TwoShotPrompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_checkpoint", default=None, type=str, required=True,
                        help="Name of model to use (e.g., 't5-base') or a path that contains the model checkpoint.")
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to a directory that contains files of the form {split}-t5.jsonl")
    parser.add_argument("--output_path", default=None, type=str, required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    chat = True
    if chat:
        prompt = ChatPrompt(few_shot_examples=TwoShotPrompt.few_shot_examples)
    else:
        prompt = TwoShotPrompt

    model = AutoregressiveModel(args.model_name_or_checkpoint, device=device)

    # Load datasets from path
    dataset_path = os.path.join(args.dataset_path, '{}-t5.jsonl')

    test_df = pd.read_json(dataset_path.format(
        'test-subsample-states'), orient='records', lines=True)

    os.makedirs(args.output_path, exist_ok=True)
    predictions_path = os.path.join(args.output_path, "predictions.jsonl")
    with open(predictions_path, "w", encoding="UTF-8") as out_f:
        for idx, ex in test_df.iterrows():
            if chat and idx % 7 == 0:
                prefix = ex["sentence_masked"].split(" <extra_id_0>")[0][:-15]
                target = ex["masked_content"].replace("<extra_id_0> ", "")
                messages = prompt.get_prompt(prefix)
                response = model.chat_generate(messages)
                out = {"input": prefix, "target": target,
                       "prediction": response}
                print(json.dumps(out), file=out_f)
                print(response)
            else:
                # TODO: implement
                continue


if __name__ == "__main__":
    main()

