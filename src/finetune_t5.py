# Code based on
# https://colab.research.google.com/drive/1eoQUsisoPmc0e-bpjSKYYd-TE1F5YTqG?usp=sharing#scrollTo=81f4PKa1F6aM
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Adafactor

from rich.table import Column, Table
from rich import box
from rich.console import Console


_MAX_SOURCE_TEXT_LENGTH = 512
_MAX_TARGET_TEXT_LENGTH = 100


class LMDataloader(Dataset):
    """Loads LM (T5 denoising) dataset with masked input."""

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_field, target_field):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_field]
        self.target_text = self.data[target_field]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Cleaning data so as to ensure data is in string type
        source_text = source_text.split()
        target_text = target_text.split()

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def display_df(df):
    """Displays dataframe in ASCII format."""

    table = Table(Column("source_text", justify="center"), Column(
        "target_text", justify="center"), title="Sample Data", pad_edge=False, box=box.ASCII)

    for _, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


def train(model, device, tokenizer,
          train_loader, train_epochs,
          optimizer, output_dir, save_every_n_epochs):

    model.train()
    for epoch in range(train_epochs):
        for step, data in enumerate(train_loader):
            labels = data['target_ids'].to(device, dtype=torch.long)
            labels[labels == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)

            loss = outputs[0]
            writer.add_scalar("Training loss", loss, step)

            if step % 100 == 0:
                training_logger.add_row(str(epoch), str(step), str(loss))
                console.print(f"Epoch {epoch}, Step: {step}, Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_every_n_epochs > 0:
            if epoch % save_every_n_epochs == 0:
                path = os.path.join(output_dir, f"model_files_ep{epoch}")
                log_path = os.path.join(output_dir, f"ep-{epoch}.log")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                console.save_text(log_path)
                console.print(f"""[Model] Model saved @ {path}\n""")


def predict(model, device, tokenizer, loader):
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    target_outputs = []
    orig_inputs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
              input_ids=ids,
              attention_mask=mask,
              max_length=_MAX_TARGET_TEXT_LENGTH,
              num_beams=3,
#              repetition_penalty=2.5,
#              length_penalty=1.0,
              early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            targs = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y]
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

            # Maybe better to save line-by-line here rather than waiting until all predictions are done?

    return predictions, target_outputs, orig_inputs, accuracy_score(target_outputs, predictions)


def T5Trainer(train_df, dev_df, test_df,
              source_field, target_field,
              output_dir,
              model,
              tokenizer,
              train_batch_size,
              valid_batch_size,
              train_epochs,
              learning_rate,
              max_source_text_length,
              max_target_text_length,
              save_every=None):
    """
    T5 trainer

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f'Device: {device}')
    model = model.to(device)

    console.log("[Data]: Reading data...\n")

    train_df = train_df[[source_field, target_field]]
    dev_df = dev_df[[source_field, target_field]]
    display_df(train_df.head(2))
    display_df(dev_df.head(2))

    console.print(f"TRAIN Dataset: {train_df.shape}")
    console.print(f"DEV Dataset: {dev_df.shape}\n")

    if test_df is not None:
        test_df = test_df[[source_field, target_field]]
        display_df(test_df.head(2))
        console.print(f"TEST Dataset: {test_df.shape}\n")

    train_dataset = LMDataloader(train_df, tokenizer, max_source_text_length,
                                 max_target_text_length, source_field, target_field)
    dev_dataset = LMDataloader(dev_df, tokenizer, max_source_text_length,
                               max_target_text_length, source_field, target_field)

    training_loader = DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_dataset, valid_batch_size, shuffle=False, num_workers=0)

    if test_df is not None:
        test_dataset = LMDataloader(test_df, tokenizer, max_source_text_length,
                                    max_target_text_length, source_field, target_field)
        test_loader = DataLoader(test_dataset, valid_batch_size, shuffle=False, num_workers=0)

    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    optimizer = Adafactor(params=model.parameters(), lr=learning_rate, relative_step=False)

    # Training loop
    console.log('[Initiating finetuning]...\n')

    train(model, device, tokenizer, training_loader, train_epochs, optimizer,
          output_dir, save_every_n_epochs=save_every)

    console.log(f'[Finished finetuning after {train_epochs} epochs.]')

    if train_epochs > 0:
        save_path = os.path.join(output_dir, "model_files")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")

        console.save_text(os.path.join(output_dir,'training_logs.txt'))
        console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'training_logs.txt')}\n""")

    console.log("[Predicting with final checkpoint]...\n")

    loaders = {
        'train': training_loader,
        'dev': val_loader,
        'test': test_loader,
    }

    eval_splits = ['dev'] if test_df is None else ['dev', 'test']

    for split in eval_splits:
        console.log(f"[Generating predictions on {split}...]\n")
        predictions, targets, orig_inputs, accuracy = predict(
            model, device, tokenizer, loaders[split])
        final_df = pd.DataFrame({'target': targets, 'prediction': predictions,'input': orig_inputs})
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'predictions_{split}.jsonl'), 'w') as wf:
            wf.write(final_df.to_json(orient='records', lines=True, force_ascii=False))
        console.print(f"""[Prediction accuracy] {accuracy}""")
        console.print(f"""Prediction data saved @ {os.path.join(output_dir)}\n""")

    console.log("[Prediction Completed.]\n")


def main(args):
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(args.seed) # pytorch random seed
    np.random.seed(args.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load datasets from path
    dataset_path = os.path.join(args.dataset_path, '{}-t5.jsonl')

    train_df = pd.read_json(dataset_path.format('train'), orient='records', lines=True)
    dev_df = pd.read_json(dataset_path.format('dev'), orient='records', lines=True)
    test_df = None

    if ('alchemy' not in args.dataset_path) and ('tw_data' not in args.dataset_path):
        test_type = 'test-disjoint-vocab' if args.test_disjoint else 'test'
        test_df = pd.read_json(dataset_path.format(test_type), orient='records', lines=True)
        dataframes = {'train_df': train_df, 'dev_df': dev_df, 'test_df': test_df}

    else:
        # Li & Andreas only test on dev so we're only using dev here, ignoring test
        dataframes = {'train_df': train_df, 'dev_df': dev_df, 'test_df': None}

    if args.prompt is not None:
        with open(args.prompt) as prompt_f:
            prompt = prompt_f.read()
        if 'zeroshot' in args.prompt:
            for df in dataframes.values():
#                df['sentence'] = prompt + ' ' + df['sentence'].astype(str)
                df['sentence_masked'] = prompt + ' ' + df['sentence_masked'].astype(str)

        elif 'incontext' in args.prompt:
            for df in dataframes.values():
                context_sents = df['sentence'].str.split(r"\. ")
                box_to_ask = context_sents.apply(lambda x: x[-1].split()[1])
                context = context_sents.apply(lambda x: '. '.join(x[:-1]) + '.')
                df['context'] = context
                df['box_to_ask'] = box_to_ask
                df['sentence_masked'] = df.apply(lambda x: prompt.format(x['context'], f'Box {x["box_to_ask"]} contains'), axis=1)
                df['masked_content'] = df['masked_content'].apply(lambda x: x.replace('<extra_id_0> ', '') + '.')
                
    console.print(train_df.sample(10))
    console.print(dev_df.sample(10))
    if test_df is not None:
        console.print(test_df.sample(10))

    # Get model parameters
    if args.model_name_or_checkpoint in ['t5-small', 't5-base', 'google/flan-t5-base']:
        model_name = args.model_name_or_checkpoint
    else:
        model_name = f'{args.model_name_or_checkpoint}/model_files_best'
    console.log(f"""[Model]: Loading {model_name}...\n""")

    if args.random_init:
        console.log(f'Using randomly initialized {model_name}.')
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM(config=config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_params = {
        'model': model,
        'tokenizer': tokenizer,
        'train_batch_size': args.train_batch_size,
        'valid_batch_size': args.val_batch_size,
        'train_epochs': args.train_epochs,
        'learning_rate': args.learning_rate,
        'max_source_text_length': _MAX_SOURCE_TEXT_LENGTH,
        'max_target_text_length': _MAX_TARGET_TEXT_LENGTH,
        'save_every': args.save_every
    }

    T5Trainer(**dataframes,
              source_field='sentence_masked',
              target_field='masked_content',
              output_dir=args.output_path,
              **model_params)


if __name__ == '__main__':
    console = Console(record=True)
    writer = SummaryWriter()
    training_logger = Table(Column("Epoch", justify="center" ),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status",pad_edge=False, box=box.ASCII)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_checkpoint', default=None, type=str, required=True,
                        help='Name of model to use (e.g., "t5-base") or a path that contains the model checkpoint.')
    parser.add_argument('--prompt', default=None, type=str)
    parser.add_argument('--dataset_path', type=str, required=True,
                            help='Path to a directory that contains files of the form {split}-t5.jsonl')
    parser.add_argument('--test_disjoint', action='store_true', help='If set, we will use test-disjoint-vocab instead of test for eval.')
    parser.add_argument('--output_path', default=None, type=str, required=True)
    parser.add_argument('--seed', default=None, type=int, required=True)
    parser.add_argument('--train_epochs', default=100, type=int, required=False)
    parser.add_argument('--early_stopping', default=False, action='store_true')
    parser.add_argument('--save_every', default=0, type=int, help='Save every n epochs.')
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--random_init', action='store_true', help='If set, we will use a randomly initialized model.') 

    args = parser.parse_args()
    console.print(args)
    main(args)
