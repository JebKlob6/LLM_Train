import logging

import datasets
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Trainer,
                          TrainingArguments)
import torch
import os

"""
      Generates TokenizeDataset
      :param table: d_name - Dataset name/path, tokenizer, Train_range, Test_range
      :return: It pre-processes and tokenizes the dataset, making it ready for training.
"""

def tokenizeDataSet(d_name, tokenizer, Train_range, Test_range):
    instruct_tune_dataset = load_dataset(d_name)
    instruct_tune_dataset = instruct_tune_dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")
    instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(Train_range))
    instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(Test_range))

    def preprocess_data(examples):
        inputs = examples['prompt']
        targets = examples['response']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = instruct_tune_dataset.map(preprocess_data, batched=True)
    return tokenized_datasets


"""
      Generates Training Args, More args could be added
      :param table: data frame with the rows `context`, `question`, and `response` and 'data_dict_form'.
      :return: appends table and adds new column based on the llm used and the results
"""


def createTrainerArgs(learning_rate, warmup_steps, evaluation_strategy,
                      per_device_train_batch_size, per_device_eval_batch_size,
                      num_train_epochs, lr_scheduler_type, **dtype_kwargs: dict):
    """
    Create a Hugging Face Trainer with specified training arguments.

    Parameters:
    - learning_rate: Initial learning rate for training.
    - warmup_steps: Number of warmup steps for learning rate scheduler.
    - evaluation_strategy: Strategy for evaluation ('steps', 'epoch').
    - per_device_train_batch_size: Batch size per device for training.
    - per_device_eval_batch_size: Batch size per device for evaluation.
    - num_train_epochs: Total number of training epochs.
    - lr_scheduler_type: Type of learning rate scheduler ('linear', 'cosine', etc.).

    Returns:
    - Trainer object ready for training.
    """
    training_args = TrainingArguments(
        output_dir='../results',
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        evaluation_strategy=evaluation_strategy,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type=lr_scheduler_type,
        logging_dir='../logs',
        **dtype_kwargs
    )
    return(training_args)

def createTrainer(model, training_args, tds, tokenizer,**dtype_kwargs: dict):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tds["train"],
        eval_dataset=tds.get("test", None),  # Use .get to avoid KeyError if 'test' is not provided
        tokenizer=tokenizer,
        **dtype_kwargs
    )

    return trainer


def train_model(model, tokenizer, trainer):
    # Train and evaluate
    logging.info("Training On Model Has Started")
    trainer.train()
    trainer.evaluate()
    logging.info("New Model Saved")
    model_path = '../trainedModels'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def train(model, dataset):
    model_name = 'google/flan-t5-base'
    data = "mosaicml/instruct-v3"
    device = 'mps'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    os.environ['export PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.1'
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    tds = tokenizeDataSet(
        "mosaicml/instruct-v3",
        tokenizer,
        1000,
        250,
    )

    args = createTrainerArgs(
        0.0005,
        35,
        'epoch',
        8,
        2,
        5,
        'linear',
    )
    TA = createTrainer(model, args, tds, tokenizer)

    train_model(model, tokenizer, TA)
