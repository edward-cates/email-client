"""Build dataset from emails with priority labels (p1, p2, p3, p4)"""
from collections import Counter
from tqdm import tqdm

from src.gmail.auth import is_authenticated
from src.gmail.config import discover_accounts
from src.gmail.service import get_all_emails_with_any_labels, get_message_count
from src.prioritization.model import get_priority_label_names, label_to_score
from src.classification.dataset import format_email_for_model


def fetch_priority_emails(limit: int | None = None) -> list[dict]:
    """Fetch ALL emails from all accounts that have any priority label (p1, p2, p3, p4).
    
    Args:
        limit: Optional maximum number of emails to fetch.
        
    Returns:
        List of email dictionaries that have exactly one priority label
    """
    priority_labels = get_priority_label_names()
    
    if not priority_labels:
        return []
    
    accounts = [acc_id for acc_id in discover_accounts() if is_authenticated(acc_id)]
    if not accounts:
        return []
    
    all_emails = []
    
    # Build Gmail query for priority labels
    query_parts = [f'label:"{name}"' for name in priority_labels]
    query = ' OR '.join(query_parts)
    
    for account_id in accounts:
        total_count = get_message_count(account_id, query=query, include_archived=True)
        
        pbar = tqdm(
            desc=f"Fetching priority emails from {account_id}",
            total=total_count,
            unit="emails",
            leave=True,
            dynamic_ncols=True
        )
        
        last_count = 0
        
        def progress_callback(current: int, total: int, acc_id: str) -> None:
            nonlocal last_count
            if current > last_count:
                last_count = current
                pbar.n = current
                pbar.refresh()
        
        account_emails = get_all_emails_with_any_labels(
            account_id,
            list(priority_labels),
            progress_callback=progress_callback
        )
        
        pbar.n = len(account_emails)
        pbar.refresh()
        pbar.close()
        
        for email in account_emails:
            email["account_id"] = account_id
        all_emails.extend(account_emails)
    
    # Filter to emails with exactly one priority label
    filtered_emails = []
    for email in all_emails:
        email_labels = set(email.get("label_names", []))
        priority_email_labels = email_labels & priority_labels
        if len(priority_email_labels) == 1:
            filtered_emails.append(email)
    
    if limit is not None:
        filtered_emails = filtered_emails[:limit]
    
    return filtered_emails


def create_huggingface_dataset(emails: list[dict] | None = None):
    """Create a HuggingFace regression dataset from emails with priority labels.
    
    Args:
        emails: Optional list of email dictionaries. If None, fetches emails using
                fetch_priority_emails().
                
    Returns:
        HuggingFace Dataset with 'text' and 'label' columns (label is float score)
    """
    from datasets import Dataset
    
    if emails is None:
        emails = fetch_priority_emails()
    
    priority_labels = get_priority_label_names()
    
    texts = []
    labels = []
    
    for email in emails:
        email_labels = set(email.get("label_names", []))
        priority_email_labels = email_labels & priority_labels
        
        assert len(priority_email_labels) == 1, (
            f"Email must have exactly 1 priority label, "
            f"found {len(priority_email_labels)}: {priority_email_labels}"
        )
        
        label_name = priority_email_labels.pop()
        score = label_to_score(label_name)
        
        text = format_email_for_model(email)
        
        texts.append(text)
        labels.append(score)
    
    return Dataset.from_dict({"text": texts, "label": labels})


if __name__ == "__main__":
    emails = fetch_priority_emails()
    print(f"Total priority emails: {len(emails)}")
    
    priority_labels = get_priority_label_names()
    
    label_counts = Counter(
        label
        for email in emails
        for label in set(email.get("label_names", [])) & priority_labels
    )
    
    print("\nPriority label counts:")
    for label in ["p1", "p2", "p3", "p4"]:
        count = label_counts.get(label, 0)
        print(f"  {label}: {count}")
    
    if emails:
        dataset = create_huggingface_dataset(emails[:5])
        print(f"\nSample dataset: {dataset}")
        print(f"Sample labels (scores): {dataset['label'][:5]}")

