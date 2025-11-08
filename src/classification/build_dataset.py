"""Build dataset from emails with custom labels"""
import json
import logging
from pathlib import Path

import yaml

from gmail.config import BASE_DIR, discover_accounts
from gmail.service import get_emails

logger = logging.getLogger(__name__)

LABELS_YAML = BASE_DIR / "src" / "gmail" / "labels.yaml"


def load_custom_labels() -> list[str]:
    """Load custom label names from labels.yaml"""
    with open(LABELS_YAML) as f:
        return [label["name"] for label in yaml.safe_load(f).get("labels", [])]


def build_label_query(labels: list[str]) -> str:
    """Build Gmail query string for labels"""
    return " OR ".join(f'label:"{label}"' for label in labels)


def fetch_emails_with_labels(account_id: str, labels: list[str], max_results: int = 1000) -> list[dict]:
    """Fetch emails that have any of the custom labels"""
    query = build_label_query(labels)
    all_emails: list[dict] = []
    page_token: str | None = None

    while len(all_emails) < max_results:
        batch_size = min(100, max_results - len(all_emails))
        emails, page_token = get_emails(account_id, max_results=batch_size, page_token=page_token, query=query)

        if not emails:
            break

        all_emails.extend(emails)

        if not page_token or len(all_emails) >= max_results:
            break

    return all_emails[:max_results]


def build_dataset(output_path: Path | str, max_emails_per_account: int = 1000) -> None:
    """Build dataset from all accounts with custom labels"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    custom_labels = load_custom_labels()
    accounts = discover_accounts()

    all_emails = []
    for account_id in accounts:
        try:
            emails = fetch_emails_with_labels(account_id, custom_labels, max_emails_per_account)
            all_emails.extend(emails)
        except Exception as e:
            logger.error("Error fetching emails for account %s: %s", account_id, e)

    # Prepare dataset
    dataset = []
    label_set = {label.lower() for label in custom_labels}

    for email in all_emails:
        email_labels = [label for label in email.get("label_names", []) if label.lower() in label_set]
        if not email_labels:
            continue

        dataset.append({
            "id": email.get("id"),
            "text": f"{email.get('subject', '')}\n{email.get('body', '')}".strip(),
            "subject": email.get("subject", ""),
            "body": email.get("body", ""),
            "from": email.get("from", ""),
            "labels": email_labels,
            "date": email.get("date", ""),
        })

    # Save dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Statistics
    label_counts: dict[str, int] = {}
    for entry in dataset:
        for label in entry["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    logger.info("Dataset saved: %d emails", len(dataset))
    for label, count in sorted(label_counts.items()):
        logger.info("  %s: %d", label, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_dataset(BASE_DIR / "data" / "email_dataset.json")

