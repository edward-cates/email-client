"""Tests for classification module"""
import sys
from unittest.mock import mock_open, patch

# Mock datasets module before any imports
class MockDataset:
    """Mock HuggingFace Dataset for testing"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, key):
        if isinstance(key, int):
            return {col: self.data[col][key] for col in self.data}
        return self.data[key]

    @staticmethod
    def from_dict(data):
        return MockDataset(data)


mock_datasets = type(sys)("datasets")
mock_datasets.Dataset = MockDataset
sys.modules["datasets"] = mock_datasets

from classification.dataset import fetch_emails_with_custom_labels


@patch("classification.dataset.get_emails")
@patch("classification.dataset.is_authenticated", return_value=True)
@patch("classification.dataset.discover_accounts")
def test_fetch_emails_until_no_labels_stops_after_10_consecutive(mock_discover, mock_auth, mock_get_emails):
    """Test that fetching stops after 10 consecutive emails without custom labels"""
    mock_discover.return_value = ["account1"]

    batch1 = [
        {"id": str(i), "label_names": ["marketing"], "internalDate": str(1000 - i)} for i in range(5)
    ] + [
        {"id": str(i), "label_names": [], "internalDate": str(1000 - i)} for i in range(5, 15)
    ]

    mock_get_emails.return_value = (batch1, None)

    result = fetch_emails_with_custom_labels()

    assert len(result) == 5  # Only emails with custom labels are returned
    assert mock_get_emails.call_count == 1


@patch("classification.dataset.get_emails")
@patch("classification.dataset.is_authenticated", return_value=True)
@patch("classification.dataset.discover_accounts")
def test_fetch_emails_until_no_labels_resets_on_label(mock_discover, mock_auth, mock_get_emails):
    """Test that counter resets when email has custom label"""
    mock_discover.return_value = ["account1"]

    batch1 = []
    for i in range(20):
        batch1.append({
            "id": str(i),
            "label_names": ["marketing"] if i % 4 == 2 else [],
            "internalDate": str(1000 - i)
        })
    batch2 = [{"id": str(i), "label_names": [], "internalDate": str(980 - i)} for i in range(20, 30)]

    mock_get_emails.side_effect = [(batch1, "token1"), (batch2, None)]

    result = fetch_emails_with_custom_labels()

    # Only emails with custom labels are returned (5 from batch1 where i % 4 == 2)
    assert len(result) == 5
    assert mock_get_emails.call_count == 2


def test_create_huggingface_dataset():
    """Test that create_huggingface_dataset creates a dataset with correct format"""
    import yaml

    # Mock labels.yaml content
    labels_content = {
        "labels": [
            {"name": "marketing", "css_class": "marketing"},
            {"name": "boring noti", "css_class": "boring-noti"},
            {"name": "Later", "css_class": "later"},
        ]
    }

    emails = [
        {
            "to": "recipient@example.com",
            "from": "sender@example.com",
            "subject": "Test Subject",
            "snippet": "This is a test email snippet.",
            "label_names": ["marketing", "CATEGORY_PROMOTIONS"],
        },
        {
            "to": "recipient2@example.com",
            "from": "sender2@example.com",
            "subject": "Another Test",
            "snippet": "Another test email snippet.",
            "label_names": ["boring noti", "CATEGORY_UPDATES"],
        },
    ]

    with patch("classification.dataset.open", mock_open(read_data=yaml.safe_dump(labels_content)), create=True):
        from classification.dataset import create_huggingface_dataset

        dataset = create_huggingface_dataset(emails)

    assert len(dataset) == 2
    assert dataset[0]["text"] == "recipient@example.com / sender@example.com / Test Subject / This is a test email snippet."
    assert dataset[0]["label_idx"] == 0
    assert dataset[1]["text"] == "recipient2@example.com / sender2@example.com / Another Test / Another test email snippet."
    assert dataset[1]["label_idx"] == 1

