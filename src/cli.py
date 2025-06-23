"""
Simple command‑line interface wrapper.

Examples
--------
Train for one epoch and save to ./ckpt:
    python -m src.cli train --epochs 1 --output_dir ./ckpt
"""

import argparse
from datasets import DatasetDict
from .data import load_imdb, preprocess, tokenizer
from .model import build_model, train as train_model, evaluate as eval_model

def parse_args():
    p = argparse.ArgumentParser(description="IMDB sentiment classifier")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="fine‑tune the model")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--output_dir", type=str, default="./checkpoints")

    e = sub.add_parser("evaluate", help="evaluate a saved checkpoint")
    e.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    if args.command == "train":
        raw = DatasetDict({
            "train": load_imdb("train"),
            "test": load_imdb("test")
        })
        tokenized = {
            "train": raw["train"].map(preprocess, batched=True),
            "test": raw["test"].map(preprocess, batched=True)
        }
        model = build_model()
        trainer = train_model(
            model,
            tokenized,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        trainer.save_model(args.output_dir)
        print("Training complete. Model saved to", args.output_dir)
    elif args.command == "evaluate":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        print("Loading model from", args.checkpoint)
        classifier = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained(args.checkpoint),
            tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
        )
        print(classifier("This movie was great!"))
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
