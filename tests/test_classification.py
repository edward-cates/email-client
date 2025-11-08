"""Tests for classification module"""
from unittest.mock import patch

import pytest

from classification.build_dataset import fetch_emails_with_custom_labels


@patch("classification.build_dataset.get_emails")
@patch("classification.build_dataset.is_authenticated", return_value=True)
@patch("classification.build_dataset.discover_accounts")
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


@patch("classification.build_dataset.get_emails")
@patch("classification.build_dataset.is_authenticated", return_value=True)
@patch("classification.build_dataset.discover_accounts")
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

