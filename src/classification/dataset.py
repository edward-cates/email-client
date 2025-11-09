"""Build dataset from emails with custom labels"""
import yaml

from src.gmail.auth import is_authenticated
from src.gmail.config import BASE_DIR, discover_accounts
from src.gmail.service import get_emails

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"


def fetch_emails_with_custom_labels(limit: int | None = None) -> list[dict]:
    """Fetch emails from all accounts until 10 consecutive have no custom labels (excluding Later),
    then filter to only return emails with at least one custom label

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
    page_tokens = dict.fromkeys(accounts, None)

    while consecutive_no_label < 10:
        if limit is not None and len(emails) >= limit:
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
            if email_labels & custom_labels:
                consecutive_no_label = 0
            else:
                consecutive_no_label += 1

            emails.append(email)

            if consecutive_no_label >= 10:
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

    Note:
        This function includes a compatibility workaround for pyarrow 22.0.0:
        The datasets library (v2.14.0) expects pyarrow.PyExtensionType, but pyarrow 22.0.0
        removed this class and only provides ExtensionType. We patch PyExtensionType to
        point to ExtensionType before importing datasets to maintain compatibility.
    """
    # Compatibility workaround: pyarrow 22.0.0 removed PyExtensionType
    # The datasets library still expects it, so we alias it to ExtensionType
    # NOTE: AI did this - it works, but seems messy.
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType"):
        pa.PyExtensionType = pa.ExtensionType  # type: ignore[attr-defined]

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

        # Format input text as "to / from / subject / snippet"
        to = email.get("to", "")
        from_addr = email.get("from", "")
        subject = email.get("subject", "")
        snippet = email.get("snippet", "").strip()

        text = f"{to} / {from_addr} / {subject} / {snippet}"

        texts.append(text)
        labels.append(label_index)

    return Dataset.from_dict({"text": texts, "label": labels})


if __name__ == "__main__":
    emails = fetch_emails_with_custom_labels()
    print(len(emails))

    dataset = create_huggingface_dataset(emails[:5])
    print(dataset)

    preview_keys = ["snippet", "subject", "from", "to", "date", "label_names"]
    email_previews = [
        {key: email.get(key, "") for key in preview_keys}
        for email in emails[:5]
    ]

    print(yaml.safe_dump(email_previews, sort_keys=False, allow_unicode=True))

