"""Gmail API service for fetching emails"""
import base64
import logging

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .auth import get_valid_credentials

logger = logging.getLogger(__name__)


def get_gmail_service(account_id: str):
    """Get authenticated Gmail service"""
    creds = get_valid_credentials(account_id)
    if not creds:
        raise ValueError(f"Account {account_id} is not authenticated")

    service = build('gmail', 'v1', credentials=creds)
    return service


def get_profile(account_id: str) -> dict:
    """Get Gmail profile information"""
    service = get_gmail_service(account_id)
    profile = service.users().getProfile(userId='me').execute()
    return profile


def list_messages(account_id: str, max_results: int = 50, query: str = None) -> list[dict]:
    """
    List messages from Gmail
    Returns list of message summaries with id, threadId
    """
    service = get_gmail_service(account_id)

    try:
        params = {
            'userId': 'me',
            'maxResults': max_results
        }
        if query:
            params['q'] = query

        results = service.users().messages().list(**params).execute()
        messages = results.get('messages', [])
        return messages
    except HttpError as error:
        logger.error("Error listing messages: %s", error)
        return []


def get_message(account_id: str, message_id: str, format: str = 'full') -> dict:
    """
    Get a full message by ID
    format: 'full', 'metadata', 'minimal', 'raw'
    """
    service = get_gmail_service(account_id)

    try:
        message = service.users().messages().get(
            userId='me',
            id=message_id,
            format=format
        ).execute()
        return message
    except HttpError as error:
        logger.error("Error getting message: %s", error)
        return {}


def parse_message(message: dict) -> dict:
    """Parse Gmail message into a simpler format"""
    payload = message.get('payload', {})
    headers = payload.get('headers', [])

    # Extract headers
    header_dict = {h['name'].lower(): h['value'] for h in headers}

    # Get body text
    body_text = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain':
                data = part.get('body', {}).get('data', '')
                if data:
                    body_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    break
    elif payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        if data:
            body_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')

    return {
        'id': message.get('id'),
        'threadId': message.get('threadId'),
        'snippet': message.get('snippet', ''),
        'subject': header_dict.get('subject', '(No subject)'),
        'from': header_dict.get('from', 'Unknown'),
        'to': header_dict.get('to', ''),
        'date': header_dict.get('date', ''),
        'body': body_text,
        'labels': message.get('labelIds', []),
        'internalDate': message.get('internalDate'),
    }


def get_emails(account_id: str, max_results: int = 50) -> list[dict]:
    """
    Get parsed emails for an account
    Returns list of parsed message dictionaries
    """
    messages = list_messages(account_id, max_results=max_results)
    emails = []

    for msg in messages:
        message = get_message(account_id, msg['id'], format='full')
        if message:
            parsed = parse_message(message)
            emails.append(parsed)

    return emails
