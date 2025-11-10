"""BERT model for email classification"""
import yaml
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.gmail.config import BASE_DIR

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"
MODEL_NAME = "distilbert-base-uncased"  # Small, fast English lowercase BERT model
MODEL_DIR = Path(__file__).parent / "model"


def load_labels() -> list[dict]:
    """Load labels from labels.yaml"""
    with open(LABELS_YAML) as f:
        return yaml.safe_load(f).get("labels", [])


def get_num_labels() -> int:
    """Get the number of labels from labels.yaml"""
    return len(load_labels())


def load_model_and_tokenizer():
    """Load BERT model and tokenizer for classification.
    
    If a trained model exists in MODEL_DIR, loads it. Otherwise loads the base model.
    
    Returns:
        tuple: (model, tokenizer) where model is a BERT model with classification head
    """
    # Check if trained model exists
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        # Load trained model (no warnings since config matches)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        # Load base model for training
        num_labels = get_num_labels()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
        )
    
    return model, tokenizer

