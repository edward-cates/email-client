"""BERT model for email classification"""
import yaml
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.gmail.config import BASE_DIR

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"
MODEL_NAME = "distilbert-base-uncased"  # Small, fast English lowercase BERT model


def load_labels() -> list[dict]:
    """Load labels from labels.yaml"""
    with open(LABELS_YAML) as f:
        return yaml.safe_load(f).get("labels", [])


def get_num_labels() -> int:
    """Get the number of labels from labels.yaml"""
    return len(load_labels())


def load_model_and_tokenizer():
    """Load BERT model and tokenizer for classification.
    
    Returns:
        tuple: (model, tokenizer) where model is a BERT model with classification head
    """
    num_labels = get_num_labels()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    
    return model, tokenizer

