"""Inference functions for email prioritization model"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.prioritization.model import load_model_and_tokenizer, MODEL_DIR, get_priority_label_names, score_to_label
from src.classification.dataset import format_email_for_model

# Global cache for model and tokenizer
_model_cache: Optional[tuple] = None


def _load_model_if_needed():
    """Load model and tokenizer, caching them for subsequent calls"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        return None
    
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    model = model.cpu()
    
    _model_cache = (model, tokenizer)
    return _model_cache


def predict_priority_score(email: dict) -> Optional[float]:
    """Predict priority score for an email.
    
    Args:
        email: Email dictionary with 'to', 'from', 'subject', 'snippet' fields
        
    Returns:
        Priority score (higher = more important, ~4.0 for p1, ~1.0 for p4),
        or None if model is not available or email already has priority label
    """
    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        return None
    
    # Check if email already has a priority label
    priority_labels = get_priority_label_names()
    email_labels = set(email.get("label_names", []))
    if email_labels & priority_labels:
        return None
    
    model_data = _load_model_if_needed()
    if model_data is None:
        return None
    
    model, tokenizer = model_data
    
    text = format_email_for_model(email)
    
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()
    
    return float(score)


def predict_priority_label(email: dict) -> Optional[str]:
    """Predict priority label for an email.
    
    Args:
        email: Email dictionary with 'to', 'from', 'subject', 'snippet' fields
        
    Returns:
        Priority label (p1, p2, p3, or p4), or None if model is not available
    """
    score = predict_priority_score(email)
    if score is None:
        return None
    
    return score_to_label(score)


def rank_emails_by_priority(emails: list[dict]) -> list[dict]:
    """Rank a list of emails by priority (highest priority first).
    
    Emails that already have priority labels are ranked by their label.
    Emails without priority labels are scored by the model.
    
    Args:
        emails: List of email dictionaries
        
    Returns:
        List of emails sorted by priority (highest first), with 'priority_score' added
    """
    priority_labels = get_priority_label_names()
    label_scores = {"p1": 4.0, "p2": 3.0, "p3": 2.0, "p4": 1.0}
    
    scored_emails = []
    
    for email in emails:
        email_copy = email.copy()
        email_labels = set(email.get("label_names", []))
        priority_email_labels = email_labels & priority_labels
        
        if priority_email_labels:
            # Use existing label
            label = priority_email_labels.pop()
            email_copy["priority_score"] = label_scores[label]
            email_copy["priority_source"] = "label"
        else:
            # Predict score
            score = predict_priority_score(email)
            if score is not None:
                email_copy["priority_score"] = score
                email_copy["priority_source"] = "predicted"
            else:
                # No model available, assign neutral score
                email_copy["priority_score"] = 2.5
                email_copy["priority_source"] = "default"
        
        scored_emails.append(email_copy)
    
    # Sort by priority score descending (higher = more important)
    scored_emails.sort(key=lambda x: x["priority_score"], reverse=True)
    
    return scored_emails

