"""
Data loading and preprocessing utilities for the IMDB sentiment dataset.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict

TOKENIZER_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def load_imdb(split: str = "train"):
    """Load a split ('train' or 'test') of the IMDB dataset."""
    return load_dataset("imdb", split=split)

def preprocess(examples: Dict, max_length: int = 256):
    """Tokenize a batch of IMDB examples (dict with key 'text')."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
