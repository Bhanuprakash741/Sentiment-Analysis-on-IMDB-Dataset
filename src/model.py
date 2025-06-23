"""
Model building, training and evaluation helpers.
"""

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import load_metric

MODEL_NAME = "distilbert-base-uncased"

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def build_model(num_labels: int = 2):
    """Return a DistilBERT model with a sequence-classification head."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}

def train(model, tokenized_datasets, output_dir: str = "./checkpoints",
          seed: int = 42, epochs: int = 3, batch_size: int = 8):
    """Fine‑tune the model and return the Trainer object."""
    set_seed(seed)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=2e-5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def evaluate(trainer):
    """Evaluate an existing Trainer object."""
    return trainer.evaluate()
