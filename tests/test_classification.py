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


@patch("classification.dataset.get_ml_label_names", return_value={"marketing", "noti"})
@patch("classification.dataset.get_all_emails_with_any_labels")
@patch("classification.dataset.is_authenticated", return_value=True)
@patch("classification.dataset.discover_accounts")
def test_fetch_emails_with_ml_labels(mock_discover, mock_auth, mock_get_all_emails, mock_get_ml_labels):
    """Test that fetching returns all emails with ML labels"""
    mock_discover.return_value = ["account1"]

    emails_with_labels = [
        {"id": str(i), "label_names": ["marketing"], "internalDate": str(1000 - i)} for i in range(5)
    ]

    mock_get_all_emails.return_value = emails_with_labels

    result = fetch_emails_with_custom_labels()

    assert len(result) == 5  # Only emails with ML labels are returned
    assert mock_get_all_emails.call_count == 1
    # Verify it was called with ML labels
    call_args = mock_get_all_emails.call_args
    assert call_args[0][0] == "account1"
    assert set(call_args[0][1]) == {"marketing", "noti"}


@patch("classification.dataset.get_ml_label_names", return_value={"marketing", "noti"})
@patch("classification.dataset.get_all_emails_with_any_labels")
@patch("classification.dataset.is_authenticated", return_value=True)
@patch("classification.dataset.discover_accounts")
def test_fetch_emails_filters_non_ml_labels(mock_discover, mock_auth, mock_get_all_emails, mock_get_ml_labels):
    """Test that emails without ML labels are filtered out"""
    mock_discover.return_value = ["account1"]

    # Mix of emails with and without ML labels
    emails = [
        {"id": "1", "label_names": ["marketing"], "internalDate": "1000"},
        {"id": "2", "label_names": ["Later"], "internalDate": "999"},  # Later has include_in_ml: false
        {"id": "3", "label_names": ["noti"], "internalDate": "998"},
        {"id": "4", "label_names": [], "internalDate": "997"},
    ]

    mock_get_all_emails.return_value = emails

    result = fetch_emails_with_custom_labels()

    # Should only return emails with ML labels (marketing and noti, not Later)
    assert len(result) == 2
    assert all("marketing" in email["label_names"] or "noti" in email["label_names"] for email in result)


def test_create_huggingface_dataset():
    """Test that create_huggingface_dataset creates a dataset with correct format"""
    import yaml

    # Mock labels.yaml content
    labels_content = {
        "labels": [
            {"name": "marketing", "include_in_ml": True},
            {"name": "noti", "include_in_ml": True},
            {"name": "Later", "include_in_ml": False},
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
            "label_names": ["noti", "CATEGORY_UPDATES"],
        },
    ]

    with patch("classification.dataset.open", mock_open(read_data=yaml.safe_dump(labels_content)), create=True):
        from classification.dataset import create_huggingface_dataset

        dataset = create_huggingface_dataset(emails)

    assert len(dataset) == 2
    assert dataset[0]["text"] == "recipient@example.com / sender@example.com / Test Subject / This is a test email snippet."
    assert dataset[0]["label"] == 0
    assert dataset[1]["text"] == "recipient2@example.com / sender2@example.com / Another Test / Another test email snippet."
    assert dataset[1]["label"] == 1

