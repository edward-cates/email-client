"""Build dataset from emails with custom labels"""
import yaml
from collections import Counter
from tqdm import tqdm

from src.gmail.auth import is_authenticated
from src.gmail.config import BASE_DIR, discover_accounts
from src.gmail.service import get_all_emails_with_any_labels, get_message_count
from src.classification.model import get_ml_label_names, load_labels

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"


def format_email_for_model(email: dict) -> str:
    """Format an email dictionary into text input for the classification model.
    
    This function converts email fields into a single text string that the model
    expects. Change this function to modify how emails are formatted for training
    and inference.
    
    Args:
        email: Email dictionary with 'to', 'from', 'subject', 'snippet' fields
        
    Returns:
        Formatted text string for model input
    """
    to = email.get("to", "")
    from_addr = email.get("from", "")
    subject = email.get("subject", "")
    snippet = email.get("snippet", "").strip()
    
    return f"{to} / {from_addr} / {subject} / {snippet}"


def fetch_emails_with_custom_labels(limit: int | None = None) -> list[dict]:
    """Fetch ALL emails from all accounts that have any ML label (where include_in_ml is True).
    
    This method uses Gmail label queries to fetch all emails with ML labels, regardless of
    when they were received or whether they're archived. This is more robust than the
    previous approach of fetching emails chronologically until finding 10 consecutive
    without labels.
    
    Args:
        limit: Optional maximum number of emails to fetch. If None, no limit is applied.
               Note: This limit is applied after fetching all emails, so it may still
               fetch all emails from Gmail before limiting.

    Returns:
        List of email dictionaries that have at least one ML label (where include_in_ml is True)
    """
    # Get ML labels (where include_in_ml is True)
    ml_labels = get_ml_label_names()
    
    if not ml_labels:
        return []

    accounts = [acc_id for acc_id in discover_accounts() if is_authenticated(acc_id)]
    if not accounts:
        return []

    all_emails = []
    
    # Build the Gmail query for all ML labels
    # Gmail query syntax: label:"name1" OR label:"name2" OR ...
    query_parts = [f'label:"{name}"' for name in ml_labels]
    query = ' OR '.join(query_parts)
    
    # Fetch all emails with any ML label from each account
    # Using get_all_emails_with_any_labels is more efficient than querying each label separately
    # as it uses a single OR query: label:"name1" OR label:"name2" OR ...
    for account_id in accounts:
        # Get total count upfront (same way inbox does it)
        total_count = get_message_count(account_id, query=query, include_archived=True)
        
        # Create a progress bar for this account with known total
        pbar = tqdm(
            desc=f"Fetching emails from {account_id}",
            total=total_count,
            unit="emails",
            leave=True,
            dynamic_ncols=True
        )
        
        # Track the last count to avoid updating too frequently
        last_count = 0
        
        def progress_callback(current: int, total: int, acc_id: str) -> None:
            """Update progress bar with current count"""
            nonlocal last_count
            # Update progress bar with the cumulative count
            # The callback from get_all_emails_with_any_labels passes (total_fetched, total_fetched, account_id)
            # which represents cumulative emails fetched across all pages
            if current > last_count:
                last_count = current
                pbar.n = current
                pbar.refresh()
        
        account_emails = get_all_emails_with_any_labels(
            account_id,
            list(ml_labels),
            progress_callback=progress_callback
        )
        
        # Update final count and close progress bar
        pbar.n = len(account_emails)
        pbar.refresh()
        pbar.close()
        
        # Add account_id to each email
        for email in account_emails:
            email["account_id"] = account_id
        all_emails.extend(account_emails)
    
    # Filter to ensure emails actually have at least one ML label
    # (Gmail queries should handle this, but this is a safety check)
    filtered_emails = []
    for email in all_emails:
        email_labels = set(email.get("label_names", []))
        if email_labels & ml_labels:
            filtered_emails.append(email)
    
    # Apply limit if specified
    if limit is not None:
        filtered_emails = filtered_emails[:limit]
    
    return filtered_emails


def create_huggingface_dataset(emails: list[dict] | None = None):
    """Create a HuggingFace classification dataset from emails with custom labels.

    Args:
        emails: Optional list of email dictionaries. If None, fetches emails using
                fetch_emails_with_custom_labels().

    Returns:
        HuggingFace Dataset with 'text' and 'label' columns

    Raises:
        AssertionError: If an email has more than one custom label (excluding non-ML labels)

    """
    # Lazy import to avoid loading datasets at module import time
    from datasets import Dataset

    if emails is None:
        emails = fetch_emails_with_custom_labels()

    labels_data = load_labels()
    custom_labels = get_ml_label_names()
    
    # Create mapping from label name to its index in the YAML array
    label_to_index = {label["name"]: idx for idx, label in enumerate(labels_data)}

    texts = []
    labels = []

    for email in emails:
        email_labels = set(email.get("label_names", []))
        custom_email_labels = email_labels & custom_labels

        assert len(custom_email_labels) == 1, (
            f"Email must have exactly 1 custom label (excluding non-ML labels), "
            f"found {len(custom_email_labels)}: {custom_email_labels}"
        )

        label_name = custom_email_labels.pop()
        label_index = label_to_index[label_name]

        # Format input text using shared function
        text = format_email_for_model(email)

        texts.append(text)
        labels.append(label_index)

    return Dataset.from_dict({"text": texts, "label": labels})


if __name__ == "__main__":
    emails = fetch_emails_with_custom_labels()
    print(len(emails))

    # Count labels
    custom_labels = get_ml_label_names()
    
    label_counts = Counter(
        label
        for email in emails
        for label in set(email.get("label_names", [])) & custom_labels
    )
    
    total = sum(label_counts.values())
    print("\nLabel counts (non-normalized):")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print("\nLabel counts (normalized):")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {label}: {percentage:.2f}%")

    dataset = create_huggingface_dataset(emails[:5])
    print(dataset)

    preview_keys = ["snippet", "subject", "from", "to", "date", "label_names"]
    email_previews = [
        {key: email.get(key, "") for key in preview_keys}
        for email in emails[:5]
    ]

    print(yaml.safe_dump(email_previews, sort_keys=False, allow_unicode=True))

