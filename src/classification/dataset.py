"""Build dataset from emails with custom labels"""
import yaml

from src.gmail.auth import is_authenticated
from src.gmail.config import BASE_DIR, discover_accounts
from src.gmail.service import get_emails

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"


def fetch_emails_with_custom_labels(limit: int | None = None) -> list[dict]:
    """Fetch emails from all accounts until 10 consecutive have no custom labels (excluding Later),
    then filter to only return emails with at least one custom label

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


if __name__ == "__main__":
    emails = fetch_emails_with_custom_labels(limit=10)
    preview_keys = ["snippet", "subject", "from", "to", "date", "label_names"]
    email_previews = [
        {key: email.get(key, "") for key in preview_keys}
        for email in emails
    ]
    print(yaml.safe_dump(email_previews, sort_keys=False, allow_unicode=True))
    print(len(emails))

