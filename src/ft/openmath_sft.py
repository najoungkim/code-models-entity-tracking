import os
from hf_olmo import *
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments

# os.environ["WANDB_PROJECT"] = "entity_tracking"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_math_dataset():
    dataset = load_dataset("nvidia/OpenMathInstruct-1")
    train_dataset = dataset["validation"]
    corr_gsm8k_train = train_dataset.filter(
        lambda x: x["is_correct"] and x["dataset"] == "gsm8k")

    print(len(corr_gsm8k_train))
    print(corr_gsm8k_train[0])

    return corr_gsm8k_train


def load_model():
    MODEL_NAME = "allenai/OLMo-1B"

    local_rank = os.getenv("LOCAL_RANK")
    device_string = "cuda:" + str(local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def get_training_args():
    # Args
    output_dir = "/home/stoshniwal/Research/entity_tracking/models/olmo_test_sft"

    per_device_train_batch_size = 8
    gradient_accumulation_steps = 128/(per_device_train_batch_size * 2)

    # Saving/Logging details
    save_steps = 100
    save_total_limit = 10
    logging_steps = 100
    num_train_epochs = 1

    # Optimizer
    optim = "adamw_torch"
    learning_rate = 2e-5
    warmup_ratio = 0.03
    lr_scheduler_type = "linear"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        # Device
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Optimizer
        optim=optim,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        bf16=True,
        warmup_ratio=warmup_ratio,
        # Save steps
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,

        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        # report_to="wandb",
        save_only_model=True,
    )

    return training_arguments


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['generated_solution'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    dataset = load_math_dataset()
    model, tokenizer = load_model()
    training_args = get_training_args()

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
        max_seq_length=1536,
    )

    trainer.train()


if __name__ == "__main__":
    main()
