import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['negative', 'neutral', 'positive']

        # Load tokenizer and model from local directory only
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path.as_posix(), local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path.as_posix(), local_files_only=True
        ).to(self.device)

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            'label': self.labels[idx],
            'confidence': float(probs[idx]),
            'probabilities': {lab: float(p) for lab, p in zip(self.labels, probs)}
        }

class EmojiPredictor:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self._load_data()

    def _load_data(self):
        emoji_file = self.data_path / "emoji_mapping.json"
        if not emoji_file.exists():
            raise FileNotFoundError(f"{emoji_file} not found")
        self.emoji_mapping = json.loads(emoji_file.read_text(encoding='utf-8'))
        self.sentiment_scores = {
            'positive': {'😊':0.95,'😃':0.92,'🥰':0.90,'😍':0.88,'🤗':0.85},
            'negative': {'😢':0.95,'😞':0.92,'😔':0.90,'💔':0.93,'😤':0.88},
            'neutral':  {'😐':0.95,'🤔':0.93,'😑':0.90,'🙄':0.85,'😶':0.88}
        }

    def predict_emojis(self, label: str, confidence: float, max_emojis: int = 5) -> List[Dict]:
        scores = self.sentiment_scores.get(label, {})
        adjusted = {e: s * confidence for e, s in scores.items()}
        top = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)[:max_emojis]
        return [{'emoji': e, 'confidence': sc} for e, sc in top]

def main():
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "data" / "models" / "sentiment"
    data_path = project_root / "data" / "processed"

    if not model_path.exists():
        print("Model directory not found. Please train the model first.")
        return

    sentiment = SentimentPredictor(str(model_path))
    emoji_predictor = EmojiPredictor(data_path)

    text = input("Enter text: ").strip()
    result = sentiment.predict(text)
    print(f"\nPredicted sentiment: {result['label']} ({result['confidence']:.2%})\n")

    emoji_list = emoji_predictor.predict_emojis(
        result['label'], result['confidence']
    )
    print("Recommended emojis:")
    for i, e in enumerate(emoji_list, 1):
        print(f" {i}. {e['emoji']} ({e['confidence']:.2%})")

if __name__ == "__main__":
    main()
