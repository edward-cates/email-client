"""Gmail API configuration"""
from pathlib import Path

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GMAIL_DIR = Path(__file__).parent
CREDENTIALS_FILE = GMAIL_DIR / "credentials.json"
TOKENS_DIR = GMAIL_DIR / "tokens"

# Create tokens directory if it doesn't exist
TOKENS_DIR.mkdir(exist_ok=True)

# OAuth redirect URI (for local server)
REDIRECT_URI = "http://localhost:9000/auth/callback"

