"""Gmail API service for fetching emails"""
import base64
import contextlib
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .auth import get_valid_credentials

logger = logging.getLogger(__name__)


def get_gmail_service(account_id: str):
    """Get authenticated Gmail service"""
    creds = get_valid_credentials(account_id)
    if not creds:
        raise ValueError(f"Account {account_id} is not authenticated")

    return build('gmail', 'v1', credentials=creds)


def get_profile(account_id: str) -> dict:
    """Get Gmail profile information"""
    service = get_gmail_service(account_id)
    profile = service.users().getProfile(userId='me').execute()
    return profile


def list_messages(account_id: str, max_results: int = 50, query: str = None, page_token: str = None) -> tuple[list[dict], str | None]:
    """List messages from Gmail inbox

    Returns:
        tuple: (messages list, next_page_token)
    """
    service = get_gmail_service(account_id)

    try:
        params = {
            'userId': 'me',
            'maxResults': max_results,
            'labelIds': ['INBOX']
        }
        if query:
            params['q'] = query
        if page_token:
            params['pageToken'] = page_token

        results = service.users().messages().list(**params).execute()
        messages = results.get('messages', [])
        next_page_token = results.get('nextPageToken')
        return messages, next_page_token
    except HttpError as error:
        logger.error("Error listing messages: %s", error)
        return [], None


def get_message_count(account_id: str, query: str = None) -> int:
    """Get total count of messages in inbox"""
    service = get_gmail_service(account_id)

    try:
        params = {
            'userId': 'me',
            'labelIds': ['INBOX'],
            'maxResults': 1  # We only need the resultSizeEstimate
        }
        if query:
            params['q'] = query

        results = service.users().messages().list(**params).execute()
        # Gmail API provides resultSizeEstimate which is approximate
        return results.get('resultSizeEstimate', 0)
    except HttpError as error:
        logger.error("Error getting message count: %s", error)
        return 0


def get_message(account_id: str, message_id: str, format: str = 'full') -> dict | None:
    """Get a full message by ID"""
    service = get_gmail_service(account_id)

    try:
        message = service.users().messages().get(
            userId='me',
            id=message_id,
            format=format
        ).execute()
        return message
    except HttpError as error:
        logger.error("Error getting message %s: %s", message_id, error)
        return None


def _extract_body_text(payload: dict) -> str:
    """Recursively extract text/plain body from message payload"""
    body_text = ""

    # Check if this part has text/plain
    if payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            with contextlib.suppress(Exception):
                body_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return body_text

    # Check nested parts
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain':
                data = part.get('body', {}).get('data', '')
                if data:
                    with contextlib.suppress(Exception):
                        body_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        if body_text:
                            break
            elif part.get('mimeType', '').startswith('multipart/'):
                # Recursively check nested multipart
                nested_text = _extract_body_text(part)
                if nested_text:
                    body_text = nested_text
                    break

    return body_text


def _extract_body_html(payload: dict) -> str:
    """Recursively extract text/html body from message payload"""
    body_html = ""

    # Check if this part has text/html
    if payload.get('mimeType') == 'text/html':
        data = payload.get('body', {}).get('data', '')
        if data:
            with contextlib.suppress(Exception):
                body_html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return body_html

    # Check nested parts
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/html':
                data = part.get('body', {}).get('data', '')
                if data:
                    with contextlib.suppress(Exception):
                        body_html = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        if body_html:
                            break
            elif part.get('mimeType', '').startswith('multipart/'):
                # Recursively check nested multipart
                nested_html = _extract_body_html(part)
                if nested_html:
                    body_html = nested_html
                    break

    return body_html


def parse_message(message: dict, label_id_to_name: dict[str, str] = None) -> dict:
    """Parse Gmail message into a simpler format"""
    payload = message.get('payload', {})
    headers = payload.get('headers', [])

    # Extract headers
    header_dict = {h['name'].lower(): h['value'] for h in headers}

    # Get body text
    body_text = _extract_body_text(payload)

    # Get label IDs and names
    label_ids = message.get('labelIds', [])
    label_names = []
    if label_id_to_name:
        label_names = [label_id_to_name.get(label_id, label_id) for label_id in label_ids]

    return {
        'id': message.get('id'),
        'threadId': message.get('threadId'),
        'snippet': message.get('snippet', ''),
        'subject': header_dict.get('subject', '(No subject)'),
        'from': header_dict.get('from', 'Unknown'),
        'to': header_dict.get('to', ''),
        'date': header_dict.get('date', ''),
        'body': body_text,
        'labels': label_ids,
        'label_names': label_names,
        'internalDate': message.get('internalDate'),
        # Note: account_id will be added by the API endpoint
    }


def parse_message_full(message: dict, label_id_to_name: dict[str, str] = None) -> dict:
    """Parse Gmail message with full details including HTML body"""
    payload = message.get('payload', {})
    headers = payload.get('headers', [])

    # Extract headers
    header_dict = {h['name'].lower(): h['value'] for h in headers}

    # Get both text and HTML body
    body_text = _extract_body_text(payload)
    body_html = _extract_body_html(payload)

    # Get label IDs and names
    label_ids = message.get('labelIds', [])
    label_names = []
    if label_id_to_name:
        label_names = [label_id_to_name.get(label_id, label_id) for label_id in label_ids]

    return {
        'id': message.get('id'),
        'threadId': message.get('threadId'),
        'snippet': message.get('snippet', ''),
        'subject': header_dict.get('subject', '(No subject)'),
        'from': header_dict.get('from', 'Unknown'),
        'to': header_dict.get('to', ''),
        'cc': header_dict.get('cc', ''),
        'bcc': header_dict.get('bcc', ''),
        'reply_to': header_dict.get('reply-to', ''),
        'date': header_dict.get('date', ''),
        'body': body_text,
        'body_html': body_html,
        'labels': label_ids,
        'label_names': label_names,
        'internalDate': message.get('internalDate'),
    }


def get_emails(account_id: str, max_results: int = 50, page_token: str = None, progress_callback: Callable[[int, int, str], None] | None = None) -> tuple[list[dict], str | None]:
    """Get parsed emails for an account with parallel fetching

    Args:
        account_id: Account identifier
        max_results: Maximum number of emails to fetch
        page_token: Pagination token
        progress_callback: Optional callback(current, total, account_id) for progress updates

    Returns:
        tuple: (emails list, next_page_token)
    """
    messages, next_page_token = list_messages(account_id, max_results=max_results, page_token=page_token)

    if not messages:
        return [], next_page_token

    # Get label ID to name mapping once for this account
    label_id_to_name = get_label_name_mapping(account_id)

    total = len(messages)
    emails = []

    # Fetch messages in parallel (max 20 concurrent requests to avoid rate limits)
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all fetch tasks
        future_to_msg = {
            executor.submit(get_message, account_id, msg['id'], 'full'): msg
            for msg in messages
        }

        # Process completed fetches
        for completed, future in enumerate(as_completed(future_to_msg), 1):
            if progress_callback:
                progress_callback(completed, total, account_id)

            try:
                message = future.result()
                if message:
                    parsed = parse_message(message, label_id_to_name)
                    emails.append(parsed)
            except Exception as e:
                logger.error("Error fetching message: %s", e)

    return emails, next_page_token


def archive_message(account_id: str, message_id: str) -> dict:
    """Archive a message by removing INBOX label"""
    service = get_gmail_service(account_id)

    try:
        result = service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': ['INBOX']}
        ).execute()
        return result
    except HttpError as error:
        logger.error("Error archiving message: %s", error)
        raise


def get_label_name_mapping(account_id: str) -> dict[str, str]:
    """Get mapping of label ID to label name for an account"""
    service = get_gmail_service(account_id)

    try:
        labels = service.users().labels().list(userId='me').execute()
        return {label['id']: label['name'] for label in labels.get('labels', [])}
    except HttpError as error:
        logger.error("Error getting labels: %s", error)
        return {}


def get_or_create_label(account_id: str, label_name: str) -> str:
    """Get existing label ID or create a new label"""
    service = get_gmail_service(account_id)

    try:
        # List all labels
        labels = service.users().labels().list(userId='me').execute()
        for label in labels.get('labels', []):
            # Case-insensitive match
            if label['name'].lower() == label_name.lower():
                return label['id']

        # Label doesn't exist, create it
        label = service.users().labels().create(
            userId='me',
            body={'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
        ).execute()
        return label['id']
    except HttpError as error:
        logger.error("Error getting/creating label %s: %s", label_name, error)
        raise


def add_labels(account_id: str, message_id: str, label_names: list[str]) -> dict:
    """Add labels to a message (creates labels if they don't exist)"""
    service = get_gmail_service(account_id)

    try:
        # Get or create label IDs
        label_ids = [get_or_create_label(account_id, name) for name in label_names]

        result = service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'addLabelIds': label_ids}
        ).execute()
        return result
    except HttpError as error:
        logger.error("Error adding labels: %s", error)
        raise


def remove_labels(account_id: str, message_id: str, label_names: list[str]) -> dict:
    """Remove labels from a message"""
    service = get_gmail_service(account_id)

    try:
        # Get label IDs (labels must exist to be removed)
        label_id_to_name = get_label_name_mapping(account_id)
        # Create case-insensitive lookup
        name_to_id = {}
        for label_id, name in label_id_to_name.items():
            name_lower = name.lower()
            if name_lower not in name_to_id:
                name_to_id[name_lower] = label_id
            # Also store original case for exact match
            if name not in name_to_id:
                name_to_id[name] = label_id

        label_ids = []
        for name in label_names:
            # Try exact match first, then case-insensitive
            if name in name_to_id:
                label_ids.append(name_to_id[name])
            elif name.lower() in name_to_id:
                label_ids.append(name_to_id[name.lower()])

        if not label_ids:
            # No labels to remove - log for debugging
            logger.warning("No matching labels found to remove for: %s", label_names)
            return {'id': message_id, 'labelIds': []}

        result = service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': label_ids}
        ).execute()
        return result
    except HttpError as error:
        logger.error("Error removing labels: %s", error)
        raise
