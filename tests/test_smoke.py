from src.data import load_imdb, preprocess
from src.model import build_model

def test_smoke():
    sample = load_imdb("train[:1]")
    tokenized = preprocess(sample)
    model = build_model()
    outputs = model(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"]
    )
    assert outputs.logits.shape[-1] == 2
