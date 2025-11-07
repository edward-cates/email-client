"""Gmail API configuration"""
from pathlib import Path

# Gmail API scopes - modify includes read access
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GMAIL_DIR = Path(__file__).parent
CREDENTIALS_DIR = GMAIL_DIR / "credentials"
TOKENS_DIR = GMAIL_DIR / "tokens"

# Create directories if they don't exist
CREDENTIALS_DIR.mkdir(exist_ok=True)
TOKENS_DIR.mkdir(exist_ok=True)

# OAuth redirect URI (for local server)
REDIRECT_URI = "http://localhost:9000/auth/callback"


def get_credentials_file(account_id: str) -> Path | None:
    """
    Get the credentials file path for an account_id.
    Maps account1 -> credentials-1.json, account2 -> credentials-2.json, etc.
    Returns None if the credentials file doesn't exist.
    """
    # Extract number from account_id (e.g., "account1" -> "1")
    try:
        account_num = account_id.replace("account", "")
        if not account_num.isdigit():
            return None
        credentials_file = CREDENTIALS_DIR / f"credentials-{account_num}.json"
        if credentials_file.exists():
            return credentials_file
    except Exception:
        pass
    return None


def discover_accounts() -> list[str]:
    """
    Discover all available accounts by checking for credentials files.
    Returns list of account_ids (e.g., ["account1", "account2", ...])
    """
    accounts = []
    for i in range(1, 10):  # Support credentials-1.json through credentials-9.json
        credentials_file = CREDENTIALS_DIR / f"credentials-{i}.json"
        if credentials_file.exists():
            accounts.append(f"account{i}")
    return sorted(accounts)
