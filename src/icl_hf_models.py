# https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelWithLMHead, LlamaForCausalLM, LlamaTokenizer

from rich.table import Column, Table
from rich import box
from rich.console import Console


_MAX_LENGTH = 512
_MAX_NEW_TOKENS = 50


class LMDataloader(Dataset):
    """Loads LM training dataset with masked input."""

    def __init__(self, dataframe, tokenizer, max_length=_MAX_LENGTH):

        self.data = dataframe
        self.tokenizer = tokenizer
        self.input_text = self.data["sentence"]
        self.prefix_text = self.data["prefix"]
        self.max_length = max_length

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])

        inp = self.tokenizer.batch_encode_plus(
            [input_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')
        
        prefix_text = str(self.prefix_text[index])
        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], padding="do_not_pad", return_tensors='pt')
        pref_lens = [len(text) for text in pref["input_ids"]]
        
        input_ids = inp['input_ids'].squeeze()
        attn_masks = inp['attention_mask'].squeeze()

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'attn_masks': attn_masks.to(dtype=torch.long),
            'prefix_lens': torch.tensor(pref_lens, dtype=torch.long),
        }

    
class LMDataloaderForInference(Dataset):
    """Loads LM dataset for inference."""

    def __init__(self, dataframe, tokenizer, max_length=_MAX_LENGTH):

        self.data = dataframe
        self.tokenizer = tokenizer
        self.prefix_text = self.data["prefix"]
        self.target_text = self.data["sentence"]
        self.max_length = max_length

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "right"
        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')

        self.tokenizer.padding_side = "left"
        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], max_length=self.max_length, pad_to_max_length=True,
            padding="max_length", return_tensors='pt')

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()

        return {
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
        }


def display_df(df):
    """Displays dataframe in ASCII format."""

    table = Table(Column("sentence", justify="center"), Column(
        "prefix", justify="center"), Column(
        "masked_content", justify="center"), title="Sample Data", pad_edge=False, box=box.ASCII)

    for _, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1], row[2])

    console.print(table)


def predict(model, device, tokenizer, loader, split, output_path):
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    target_outputs = []
    orig_inputs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            targets = data['target_ids'].to(device, dtype = torch.long)
            ids = data['prefix_ids'].to(device, dtype = torch.long)
            mask = data['prefix_attn_masks'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids=ids,
              attention_mask=mask,
              max_new_tokens=_MAX_NEW_TOKENS,
              num_beams=3,
#              repetition_penalty=2.5,
#              length_penalty=1.0,
              early_stopping=True
            )

            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            targs = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in targets]
            inputs = [tokenizer.decode(
                inp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for inp in ids]

            if i % 10==0:
                console.print(f'Completed {i}\n')
                assert len(preds) == len(targs) == len(inputs)
                console.print('target\tpredicted\tinput\n')
                for pred, target, inp in zip(preds, targs, inputs):
                    console.print(f'{target}\t{pred}\t{inp}\n')
            predictions.extend(preds)
            target_outputs.extend(targs)
            orig_inputs.extend(inputs)

            with open(output_path, 'a') as wf:
                for pred, target, inp in zip(preds, targs, inputs):
                    pred_w = pred.removeprefix(inp + " ")
                    targ_w = target.removeprefix(inp + " ").strip(".")
                    inp_w = inp + ' .'
                    wf.write(json.dumps({'target': targ_w, 'prediction': pred_w, 'input': inp_w}) + '\n')

    return predictions, target_outputs, orig_inputs, accuracy_score(target_outputs, predictions)


def main(args):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    with open(args.dataset_path) as f:
        lines = f.readlines()
    line_dicts = [json.loads(line) for line in lines[:3308]]
    eval_df = pd.DataFrame(line_dicts)
    if args.prompt is not None:
        with open(args.prompt) as prompt_f:
            prompt = prompt_f.read()
        context_sents = eval_df['prefix'].str.split(r"\. ")
        box_to_ask = context_sents.apply(lambda x: x[-1].split()[1])
        eval_df['box_to_ask'] = box_to_ask
        eval_df['prompt'] = eval_df.apply(lambda x: prompt.format(desc=x['prefix'], box_no=x["box_to_ask"]), axis=1)

    console.print(eval_df.sample(10))

    # Get model parameters
    model_name = args.model_name_or_checkpoint
    console.log(f"""[Model]: Loading {model_name}...\n""")

    if 'goat' in model_name:
        # not sure if this is a legit model
        model = LlamaForCausalLM.from_pretrained("baffo32/decapoda-research-llama-7B-hf")
        # model = LlamaForCausalLM.from_pretrained("yahma/llama-7b-hf")
        model = PeftModel.from_pretrained(model, model_name)
        tokenizer_left = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer', legacy=False)
        # tokenizer_left = LlamaTokenizer.from_pretrained("yahma/llama-7b-hf")
        # decapoda issue
        model.config.pad_token_id = tokenizer_left.pad_token_id = 0
        model.config.bos_token_id = tokenizer_left.bos_token_id = 1
        model.config.eos_token_id = tokenizer_left.eos_token_id = 2
    else:
        model = AutoModelWithLMHead.from_pretrained(model_name)
        tokenizer_left = AutoTokenizer.from_pretrained(model_name)
        tokenizer_left.pad_token = tokenizer_left.eos_token
        tokenizer_left.pad_token_id = tokenizer_left.eos_token_id

    tokenizer_left.padding_side = "left"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f'Device: {device}')
    model = model.to(device)

    console.log("[Data]: Reading data...\n")
    display_df(eval_df.head(2))
    console.print(f"TEST Dataset: {eval_df.shape}\n")    
    test_dataset = LMDataloaderForInference(eval_df, tokenizer_left, _MAX_LENGTH)
    test_loader = DataLoader(test_dataset, args.val_batch_size, shuffle=False, num_workers=0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    console.log(f"[Generating predictions...]\n")
    _, _, _, accuracy = predict(
        model, device, tokenizer_left, test_loader, 'test', args.output_path)
    console.print(f"""[Prediction accuracy] {accuracy}""")
    console.print(f"""Prediction data saved @ {os.path.join(args.output_path)}\n""")
    console.log("[Prediction Completed.]\n")


if __name__ == '__main__':
    console = Console(record=True)
    writer = SummaryWriter()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_checkpoint', default=None, type=str, required=True,
                        help='Name of model to use (e.g., "t5-base") or a path that contains the model checkpoint.')
    parser.add_argument('--prompt', default=None, type=str)
    parser.add_argument('--dataset_path', type=str, required=True,
                            help='Path to a directory that contains files of the form {split}-t5.jsonl')
    parser.add_argument('--test_disjoint', action='store_true', help='If set, we will use test-disjoint-vocab instead of test for eval.')
    parser.add_argument('--output_path', default=None, type=str, required=True)
    parser.add_argument('--seed', default=None, type=int, required=True)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--random_init', action='store_true', help='If set, we will use a randomly initialized model.') 
    parser.add_argument('--condensed', action='store_true', help='If set, we will use the version of the dataset that predicts all box states in one example.')

    args = parser.parse_args()
    console.print(args)
    main(args)
