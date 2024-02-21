import datasets
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, AutoModel,
                          TrainingArguments)
import torch
import os

# Load and preprocess dataset
instruct_tune_dataset = load_dataset("mosaicml/instruct-v3")
instruct_tune_dataset = instruct_tune_dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")
instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(500))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(50))

tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

def preprocess_data(examples):
    inputs = examples['prompt']
    targets = examples['response']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = instruct_tune_dataset.map(preprocess_data, batched=True)

# Setup for MPS device
device = torch.device('mps')

# Initialize model and move it to MPS
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(device)

os.environ['export PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")
# Train and evaluate
trainer.train()
trainer.evaluate()
