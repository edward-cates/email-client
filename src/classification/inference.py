"""Inference functions for email classification model"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.classification.model import load_model_and_tokenizer, MODEL_DIR, load_labels, get_ml_label_names
from src.classification.dataset import format_email_for_model
from src.gmail.config import BASE_DIR

MODEL_WEIGHTS_PATH = MODEL_DIR  # For backward compatibility check
LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"

# Global cache for model and tokenizer
_model_cache: Optional[tuple] = None


def _load_model_if_needed():
    """Load model and tokenizer, caching them for subsequent calls"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    # Check if trained model exists (either new format or old format for backward compatibility)
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        # Check for old format
        old_weights_path = Path(__file__).parent / "model_weights.pt"
        if not old_weights_path.exists():
            return None
        # Old format - load base model and load weights
        num_labels = len(load_labels())
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
        )
        model.load_state_dict(torch.load(old_weights_path, map_location='cpu'))
    else:
        # New format - load using transformers' from_pretrained
        model, tokenizer = load_model_and_tokenizer()
    
    model.eval()
    # Ensure model is on CPU for inference
    model = model.cpu()
    
    _model_cache = (model, tokenizer)
    return _model_cache


def predict_email_labels(email: dict) -> Optional[dict[str, int]]:
    """Predict label scores for an email using the trained model.
    
    Args:
        email: Email dictionary with 'to', 'from', 'subject', 'snippet' fields
        
    Returns:
        Dictionary mapping label names to scores (0-100, rounded down to whole percent),
        or None if model is not available or email has custom labels
    """
    # Check if model exists (new format or old format)
    model_exists = (MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()) or \
                   (Path(__file__).parent / "model_weights.pt").exists()
    if not model_exists:
        return None
    
    # Load labels and check if email has custom labels (excluding non-ML labels)
    labels_data = load_labels()
    custom_labels = get_ml_label_names()
    
    email_labels = set(email.get("label_names", []))
    has_custom_label = bool(email_labels & custom_labels)
    
    # Only predict for emails without custom labels
    if has_custom_label:
        return None
    
    # Load model if needed
    model_data = _load_model_if_needed()
    if model_data is None:
        return None
    
    model, tokenizer = model_data
    
    # Format input text using shared function
    text = format_email_for_model(email)
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
    
    # Get label names and indices for ML labels only
    label_names = [label["name"] for label in labels_data if label.get("include_in_ml", True)]
    label_indices = [i for i, label in enumerate(labels_data) if label.get("include_in_ml", True)]
    
    # Create dictionary mapping label names to scores (rounded down to whole percent)
    # Use lowercase keys for case-insensitive lookup
    scores = {}
    for label_name, label_idx in zip(label_names, label_indices):
        # Convert probability to percentage and round down
        score = int(np.floor(probs[label_idx] * 100))
        scores[label_name.lower()] = score
    
    return scores

