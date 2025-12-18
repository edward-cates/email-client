"""BERT model for email prioritization (regression)"""
from pathlib import Path

from src.gmail.config import BASE_DIR

# Priority labels and their numeric scores (higher = more important)
PRIORITY_LABELS = ["p1", "p2", "p3", "p4"]
PRIORITY_SCORES = {"p1": 4.0, "p2": 3.0, "p3": 2.0, "p4": 1.0}

MODEL_NAME = "distilbert-base-uncased"  # Same base model as classification
MODEL_DIR = Path(__file__).parent / "model"


def get_priority_label_names() -> set[str]:
    """Get set of priority label names (p1, p2, p3, p4)"""
    return set(PRIORITY_LABELS)


def label_to_score(label: str) -> float:
    """Convert priority label to numeric score.
    
    Args:
        label: Priority label (p1, p2, p3, p4)
        
    Returns:
        Numeric score (4.0 for p1, 3.0 for p2, 2.0 for p3, 1.0 for p4)
    """
    return PRIORITY_SCORES[label]


def score_to_label(score: float) -> str:
    """Convert numeric score back to priority label (for evaluation).
    
    Uses rounding to nearest priority level.
    
    Args:
        score: Numeric score
        
    Returns:
        Closest priority label
    """
    rounded = round(score)
    rounded = max(1, min(4, rounded))  # Clamp to valid range
    score_to_label_map = {4: "p1", 3: "p2", 2: "p3", 1: "p4"}
    return score_to_label_map[rounded]


def load_model_and_tokenizer():
    """Load BERT model and tokenizer for regression.
    
    If a trained model exists in MODEL_DIR, loads it. Otherwise loads the base model.
    
    Returns:
        tuple: (model, tokenizer) where model is a BERT model with regression head (num_labels=1)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Check if trained model exists
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        # Load base model for training - num_labels=1 for regression
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            problem_type="regression",
        )
    
    return model, tokenizer

