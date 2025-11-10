"""BERT model for email classification"""
import yaml
from pathlib import Path

from src.gmail.config import BASE_DIR

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"
MODEL_NAME = "distilbert-base-uncased"  # Small, fast English lowercase BERT model
MODEL_DIR = Path(__file__).parent / "model"


def load_labels() -> list[dict]:
    """Load labels from labels.yaml"""
    with open(LABELS_YAML) as f:
        return yaml.safe_load(f).get("labels", [])


def get_ml_labels() -> list[dict]:
    """Get labels that should be included in ML training and inference.
    
    Returns:
        List of label dictionaries where include_in_ml is True (defaults to True if not specified)
    """
    labels = load_labels()
    return [label for label in labels if label.get("include_in_ml", True)]


def get_ml_label_names() -> set[str]:
    """Get set of label names that should be included in ML.
    
    Returns:
        Set of label names where include_in_ml is True
    """
    return {label["name"] for label in get_ml_labels()}


def get_num_labels() -> int:
    """Get the number of labels from labels.yaml"""
    return len(load_labels())


def load_model_and_tokenizer():
    """Load BERT model and tokenizer for classification.
    
    If a trained model exists in MODEL_DIR, loads it. Otherwise loads the base model.
    
    Returns:
        tuple: (model, tokenizer) where model is a BERT model with classification head
    """
    # Lazy import to avoid importing transformers at module level (helps with testing)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
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

