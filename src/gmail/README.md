# Gmail API Setup

## Prerequisites

1. Create a Google Cloud Project:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. Enable Gmail API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click "Enable"

3. Create OAuth 2.0 Credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - If prompted, configure the OAuth consent screen:
     - Choose "External" user type
     - Fill in app name, user support email, developer contact
     - Add your email to test users
   - For application type, choose "Desktop app"
   - Click "Create"
   - Download the credentials JSON file

4. Place credentials file:
   - Rename the downloaded file to `credentials.json`
   - Place it in the `src/gmail/` directory

## Directory Structure

After setup, your `src/gmail/` directory should contain:
- `credentials.json` - Your OAuth credentials (downloaded from Google Cloud)
- `tokens/` - Directory for storing authentication tokens (created automatically)
  - `token_account1.json` - Token for first account
  - `token_account2.json` - Token for second account

## Usage

1. Start the server: `python src/web/main.py`
2. Open `http://localhost:9000` in your browser
3. Click "Sign in with Google" for each account you want to add
4. Complete the OAuth flow in your browser
5. Your emails will be displayed automatically

## Security Note

The `tokens/` directory contains sensitive authentication tokens. Keep this directory secure and don't commit it to version control. Consider adding `tokens/` to your `.gitignore`.

