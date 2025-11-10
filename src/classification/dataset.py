"""Build dataset from emails with custom labels"""
import yaml
from collections import Counter

from src.gmail.auth import is_authenticated
from src.gmail.config import BASE_DIR, discover_accounts
from src.gmail.service import get_emails

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
    """Fetch emails from all accounts until 10 consecutive have no custom labels (excluding Later),
    then filter to only return emails with at least one custom label

    Logic: (1) Keep fetching until finding at least one email with a custom label,
           (2) Then stop once 10 consecutive emails without custom labels are found.

    Uses pagination to keep getting emails.

    Args:
        limit: Optional maximum number of emails to fetch. If None, no limit is applied.

    Returns:
        List of email dictionaries that have at least one custom label
    """
    with open(LABELS_YAML) as f:
        custom_labels = {label["name"] for label in yaml.safe_load(f).get("labels", [])}
    custom_labels.discard("Later")

    accounts = [acc_id for acc_id in discover_accounts() if is_authenticated(acc_id)]
    if not accounts:
        return []

    emails = []
    consecutive_no_label = 0
    found_labeled_email = False  # Track if we've found at least one email with a custom label
    page_tokens = dict.fromkeys(accounts, None)

    while True:
        if limit is not None and len(emails) >= limit:
            break

        # If we've found at least one labeled email and hit 10 consecutive without labels, stop
        if found_labeled_email and consecutive_no_label >= 10:
            break

        batch = []
        next_tokens = {}

        for account_id in accounts:
            acc_emails, next_token = get_emails(account_id, max_results=50, page_token=page_tokens.get(account_id), include_archived=True)
            for email in acc_emails:
                email["account_id"] = account_id
            batch.extend(acc_emails)
            if next_token:
                next_tokens[account_id] = next_token

        if not batch:
            break

        batch.sort(key=lambda x: int(x.get("internalDate", 0) or 0), reverse=True)

        for email in batch:
            if limit is not None and len(emails) >= limit:
                break

            email_labels = set(email.get("label_names", []))
            has_custom_label = bool(email_labels & custom_labels)
            
            if has_custom_label:
                found_labeled_email = True
                consecutive_no_label = 0
            else:
                # Only count consecutive if we've already found at least one labeled email
                if found_labeled_email:
                    consecutive_no_label += 1

            emails.append(email)

            # Stop if we've found labeled emails and hit 10 consecutive without
            if found_labeled_email and consecutive_no_label >= 10:
                break

        page_tokens = next_tokens
        if not page_tokens:
            break

    # Filter to only return emails with at least one custom label
    filtered_emails = []
    for email in emails:
        email_labels = set(email.get("label_names", []))
        if email_labels & custom_labels:
            filtered_emails.append(email)

    return filtered_emails


def create_huggingface_dataset(emails: list[dict] | None = None):
    """Create a HuggingFace classification dataset from emails with custom labels.

    Args:
        emails: Optional list of email dictionaries. If None, fetches emails using
                fetch_emails_with_custom_labels().

    Returns:
        HuggingFace Dataset with 'text' and 'label' columns

    Raises:
        AssertionError: If an email has more than one custom label (excluding "Later")

    """
    # Lazy import to avoid loading datasets at module import time
    from datasets import Dataset

    if emails is None:
        emails = fetch_emails_with_custom_labels()

    with open(LABELS_YAML) as f:
        labels_data = yaml.safe_load(f).get("labels", [])
        custom_labels = {label["name"] for label in labels_data}
        custom_labels.discard("Later")
        
        # Create mapping from label name to its index in the YAML array
        label_to_index = {label["name"]: idx for idx, label in enumerate(labels_data)}

    texts = []
    labels = []

    for email in emails:
        email_labels = set(email.get("label_names", []))
        custom_email_labels = email_labels & custom_labels

        assert len(custom_email_labels) == 1, (
            f"Email must have exactly 1 custom label (excluding 'Later'), "
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
    with open(LABELS_YAML) as f:
        custom_labels = {label["name"] for label in yaml.safe_load(f).get("labels", [])}
    custom_labels.discard("Later")
    
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

