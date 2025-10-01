import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir.as_posix(), local_files_only=True
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return tokenizer, model

def predict_sentiment(tokenizer, model, text: str):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    labels = ['negative', 'neutral', 'positive']
    idx = int(probs.argmax())
    return labels[idx], float(probs[idx])

def main():
    # Locate project root and model path
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "data" / "models" / "sentiment"
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    print("Loading model...")
    tokenizer, model = load_model(model_dir)
    print("✅ Model loaded.")

    # Interactive prompt
    text = input("Enter text: ").strip()
    if not text:
        print("No text entered. Exiting.")
        return

    start = time.time()
    label, confidence = predict_sentiment(tokenizer, model, text)
    elapsed = time.time() - start

    print(f"\nPredicted Sentiment: {label.upper()} ({confidence:.1%})")
    print(f"Inference Time: {elapsed*1000:.0f} ms")

if __name__ == "__main__":
    main()
