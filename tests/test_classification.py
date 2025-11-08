"""Tests for classification module"""
from unittest.mock import patch

import pytest

from classification.build_dataset import build_label_query, fetch_emails_with_labels


class TestBuildLabelQuery:
    """Tests for build_label_query function"""

    def test_single_label(self):
        """Test query building with single label"""
        result = build_label_query(["marketing"])
        assert result == 'label:"marketing"'

    def test_multiple_labels(self):
        """Test query building with multiple labels"""
        result = build_label_query(["marketing", "newsletter", "event"])
        assert result == 'label:"marketing" OR label:"newsletter" OR label:"event"'

    def test_empty_labels(self):
        """Test query building with empty list"""
        result = build_label_query([])
        assert result == ""


class TestFetchEmailsWithLabels:
    """Tests for fetch_emails_with_labels function"""

    @patch("classification.build_dataset.get_emails")
    def test_single_page(self, mock_get_emails):
        """Test fetching emails from single page"""
        mock_emails = [
            {"id": "1", "subject": "Test 1", "label_names": ["marketing"]},
            {"id": "2", "subject": "Test 2", "label_names": ["newsletter"]},
        ]
        mock_get_emails.return_value = (mock_emails, None)

        result = fetch_emails_with_labels("account1", ["marketing", "newsletter"], max_results=100)

        assert len(result) == 2
        assert result == mock_emails
        mock_get_emails.assert_called_once_with(
            "account1", max_results=100, page_token=None, query='label:"marketing" OR label:"newsletter"'
        )

    @patch("classification.build_dataset.get_emails")
    def test_pagination(self, mock_get_emails):
        """Test fetching emails with pagination"""
        page1 = [{"id": str(i), "subject": f"Test {i}", "label_names": ["marketing"]} for i in range(100)]
        page2 = [{"id": str(i), "subject": f"Test {i}", "label_names": ["newsletter"]} for i in range(100, 150)]

        mock_get_emails.side_effect = [
            (page1, "token1"),
            (page2, None),
        ]

        result = fetch_emails_with_labels("account1", ["marketing"], max_results=200)

        assert len(result) == 150
        assert mock_get_emails.call_count == 2
        mock_get_emails.assert_any_call(
            "account1", max_results=100, page_token=None, query='label:"marketing"'
        )
        mock_get_emails.assert_any_call(
            "account1", max_results=100, page_token="token1", query='label:"marketing"'
        )

    @patch("classification.build_dataset.get_emails")
    def test_max_results_limit(self, mock_get_emails):
        """Test that max_results limit is respected"""
        page1 = [{"id": str(i), "subject": f"Test {i}", "label_names": ["marketing"]} for i in range(100)]
        page2 = [{"id": str(i), "subject": f"Test {i}", "label_names": ["marketing"]} for i in range(100, 200)]

        mock_get_emails.side_effect = [
            (page1, "token1"),
            (page2, "token2"),
        ]

        result = fetch_emails_with_labels("account1", ["marketing"], max_results=150)

        assert len(result) == 150
        assert mock_get_emails.call_count == 2

    @patch("classification.build_dataset.get_emails")
    def test_empty_results(self, mock_get_emails):
        """Test handling of empty results"""
        mock_get_emails.return_value = ([], None)

        result = fetch_emails_with_labels("account1", ["marketing"], max_results=100)

        assert len(result) == 0
        assert result == []
        mock_get_emails.assert_called_once()

    @patch("classification.build_dataset.get_emails")
    def test_small_batch_size(self, mock_get_emails):
        """Test that batch size is correctly calculated for small max_results"""
        mock_emails = [{"id": "1", "subject": "Test", "label_names": ["marketing"]}]
        mock_get_emails.return_value = (mock_emails, None)

        result = fetch_emails_with_labels("account1", ["marketing"], max_results=50)

        assert len(result) == 1
        mock_get_emails.assert_called_once_with(
            "account1", max_results=50, page_token=None, query='label:"marketing"'
        )

