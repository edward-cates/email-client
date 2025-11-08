"""Gmail OAuth 2.0 authentication"""
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .config import REDIRECT_URI, SCOPES, TOKENS_DIR, get_credentials_file


def get_token_path(account_id: str) -> Path:
    """Get the token file path for an account"""
    return TOKENS_DIR / f"token_{account_id}.json"


def load_credentials(account_id: str) -> Credentials | None:
    """Load credentials from token file"""
    token_path = get_token_path(account_id)
    if not token_path.exists():
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        return creds
    except (ValueError, OSError, KeyError):
        return None


def save_credentials(account_id: str, creds: Credentials) -> None:
    """Save credentials to token file"""
    token_path = get_token_path(account_id)
    with open(token_path, 'w') as token:
        token.write(creds.to_json())


def refresh_credentials(creds: Credentials) -> Credentials:
    """Refresh expired credentials"""
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def get_authorization_url(account_id: str) -> tuple[str, str]:
    """
    Start OAuth flow and return authorization URL and state
    Returns: (authorization_url, state)
    """
    credentials_file = get_credentials_file(account_id)
    if not credentials_file:
        raise FileNotFoundError(
            f"Credentials file not found for {account_id}. "
            f"Please add credentials-{account_id.replace('account', '')}.json to src/gmail/credentials/"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_file), SCOPES
    )
    flow.redirect_uri = REDIRECT_URI

    # Generate state to track account_id
    state = account_id

    authorization_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=state
    )

    return authorization_url, state


def complete_authorization(code: str, state: str) -> Credentials:
    """
    Complete OAuth flow with authorization code
    Returns: Credentials object
    """
    account_id = state
    credentials_file = get_credentials_file(account_id)
    if not credentials_file:
        raise FileNotFoundError(f"Credentials file not found for {account_id}")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_file), SCOPES
    )
    flow.redirect_uri = REDIRECT_URI

    flow.fetch_token(code=code)
    creds = flow.credentials

    # Save credentials using state as account_id
    # Type check: flow.credentials can be either Credentials type, but save_credentials
    # expects google.oauth2.credentials.Credentials. In practice, OAuth flow returns
    # google.oauth2.credentials.Credentials, so we type cast it.
    save_credentials(account_id, creds)  # type: ignore[arg-type]

    return creds  # type: ignore[return-value]


def get_valid_credentials(account_id: str) -> Credentials | None:
    """
    Get valid credentials for an account, refreshing if needed
    Returns None if not authenticated
    """
    creds = load_credentials(account_id)
    if not creds:
        return None

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds = refresh_credentials(creds)
                save_credentials(account_id, creds)
            except (ValueError, OSError):
                return None
        else:
            return None

    return creds


def is_authenticated(account_id: str) -> bool:
    """Check if account is authenticated"""
    creds = get_valid_credentials(account_id)
    return creds is not None


def get_account_email(account_id: str) -> str | None:
    """Get the email address for an authenticated account"""
    try:
        from .service import get_profile
        profile = get_profile(account_id)
        return profile.get('emailAddress')
    except (ValueError, KeyError):
        return None
