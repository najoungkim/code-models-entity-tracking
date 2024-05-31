import argparse
import json
import pandas as pd
import time
import os
import requests
from transformers import AutoTokenizer






from prompt.base_prompt import ChatPrompt
from prompt.prompt_library import TwoShotPrompt

NUM_BOXES = 7

def generate_hf_hub(args, input):
    payload = {
        "inputs": input,
        "parameters": {
            "temperature": 0.01,
            "max_new_tokens": args.max_tokens,
            "do_sample": False,
            "return_text": True,
            "return_tensors": False ,
            "stop": ["\n"] if not args.chat else []
        }
    }
    
    headers = {
      "Accept" : "application/json",
      "Authorization": f"Bearer {args.hf_token}",
      "Content-Type": "application/json"
    }
    
    response = requests.post(args.hf_endpoint, headers=headers, json=payload)
    return response.json()[0]["generated_text"]






def main(args):

    chat = args.chat
    tokenizer = None
    if chat:
        prompt = ChatPrompt(few_shot_examples=TwoShotPrompt.few_shot_examples)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_checkpoint)
    else:
        prompt = TwoShotPrompt

    dataset_path = os.path.join(args.dataset_path, '{}-t5.jsonl')
    test_df = pd.read_json(dataset_path.format(
        'test-subsample-states'), orient='records', lines=True)

    if args.num_examples > -1:
        test_df = test_df.head(n=args.num_examples)

    os.makedirs(args.output_path, exist_ok=True)
    predictions_path = os.path.join(args.output_path, "predictions.jsonl")
    with open(predictions_path, "w", encoding="UTF-8") as out_f:
        for idx, ex in test_df.iterrows():
            # test in condensed format, so only consider every BOX_NUMBERth entry
            if idx % NUM_BOXES == 0:
                prefix = ex["sentence_masked"].split(" <extra_id_0>")[0][:-15]
                target = ex["masked_content"].replace("<extra_id_0> ", "")
                if chat:
                    messages = prompt.get_prompt(prefix)
                    messages[0]["content"] = "You are a helpful assistant. Given the description after \"Description:\", write a true statement about the contents of all the boxes according to the description after \"Statement:\". Make sure to always output all seven boxes in one line and make sure that you use the same format in every response."
                    input = tokenizer.apply_chat_template(messages, tokenize=False) + " Box 0 contains"
                    #print(input)

                    #response = model.chat_generate(messages)
                else:
                    input = prompt.get_prompt(prefix) + " Box 0 contains"
                
                response = generate_hf_hub(args, input)
                
                # TODO: target is wrong since it only outputs one box
                # this is currently taken care of by expand_results.py
                # but would be cleaner to fix this here
                out = {"input": prefix, "target": target,
                       "prediction": response}
                print(json.dumps(out), file=out_f)
                print(idx // NUM_BOXES)
                out_f.flush()



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_checkpoint", default=None, type=str, required=True,
                        help="Name of model to use (e.g., 't5-base') or a path that contains the model checkpoint.")

    parser.add_argument('--dataset_path', type=str, required=True,
                            help='Path to a directory that contains files of the form {split}-t5.jsonl')
    parser.add_argument('--output_path', default=None, type=str, required=True)
    

    parser.add_argument('--num_examples', default=-1, type=int,
                        help="Number of example to run. Set to -1 to run all examples.")
    
    parser.add_argument('--chat', action="store_true",
                        help="If set, use a chat template.")
    
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help="Maximum number of generated tokens (default: 512).")

    parser.add_argument('--hf_token',
                        type=str,
                        help="HuggingFace access token")

    parser.add_argument('--hf_endpoint',
                        type=str,
                        help="URL to HuggingFace Inference endpoint")

    args = parser.parse_args()
    main(args)



