import replicate
import argparse
import json
import pandas as pd
import time
import os


from replicate.exceptions import ReplicateException

from prompt.base_prompt import ChatPrompt
from prompt.prompt_library import TwoShotPrompt

NUM_BOXES = 7

def generate_replicate(args, input):
    input_c =  {
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": input,
        "temperature": 0.01,
        "max_new_tokens": args.max_tokens,
        "min_new_tokens": -1,
        "stop_sequences": "\n",
        "stop_str": "\n"
    }

    tries = 0
    while tries < 5:
        tries += 1
        try:
            response = replicate.run(args.model_name, input=input_c)
            break
        except ReplicateException as e:
            print(e)
            print("API Error: retrying in 1min")
            time.sleep(60.0)
    
    return "".join(response)







def main(args):

    chat = args.chat
    if chat:
        prompt = ChatPrompt(few_shot_examples=TwoShotPrompt.few_shot_examples)
        raise NotImplementedError
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
                    raise NotImplementedError
                    #response = model.chat_generate(messages)
                else:
                    input = prompt.get_prompt(prefix) + " Box 0 contains"
                    response = generate_replicate(args, input)
                
                # TODO: target is wrong since it only outputs one box
                # this is currently taken care of by expand_results.py
                # but would be cleaner to fix this here
                out = {"input": prefix, "target": target,
                       "prediction": response}
                print(json.dumps(out), file=out_f)
                out_f.flush()



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default="meta/llama-2-70b", type=str,
                        help='Name of GPT-3 model (default: meta/llama-2-70b')
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

    args = parser.parse_args()
    main(args)



