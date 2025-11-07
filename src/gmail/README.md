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
   - You might need to create a project, not sure (already had one).
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - For application type, choose "Desktop app"
   - Click "Create"
   - Download the credentials JSON file

4. Place credentials file(s):
   - For a single account: Rename the downloaded file to `credentials-1.json` and place it in `src/gmail/credentials/`
   - For multiple accounts: Repeat step 3 for each account, naming them `credentials-1.json`, `credentials-2.json`, etc. (up to `credentials-9.json`)
   - Each credentials file corresponds to one Gmail account

## Multi-Account Setup

This client supports up to 9 Gmail accounts in a merged inbox:

1. **Add credential files**: Place `credentials-1.json` through `credentials-9.json` in `src/gmail/credentials/`
   - Each file should be the OAuth credentials for a different Gmail account
   - You can add accounts incrementally - just add the credential file and restart the server

2. **Authenticate accounts**: 
   - Start the server and open `http://localhost:9000`
   - The sidebar will show all discovered accounts
   - Click "Sign in" next to any unauthenticated account
   - Complete the OAuth flow in your browser

3. **Merged inbox**: Once authenticated, emails from all accounts are automatically merged into a single inbox, sorted by date (newest first)

4. **Account identification**: Each email shows which account it belongs to (1-9), and actions like archiving or adding labels are performed on the correct account automatically

## Directory Structure

After setup, your `src/gmail/` directory should contain:
- `credentials/` - Directory for OAuth credential files
  - `credentials-1.json` - OAuth credentials for account1
  - `credentials-2.json` - OAuth credentials for account2 (optional)
  - ... up to `credentials-9.json` (optional)
- `tokens/` - Directory for storing authentication tokens (created automatically)
  - `token_account1.json` - Token for account1 (created after authentication)
  - `token_account2.json` - Token for account2 (created after authentication)
  - ... etc.

## Usage

1. Start the server: `python src/web/main.py` or `make up`
2. Open `http://localhost:9000` in your browser
3. The sidebar will show all accounts found in `src/gmail/credentials/`
4. Click "Sign in" next to any account you want to authenticate
5. Complete the OAuth flow in your browser
6. Your emails from all authenticated accounts will be displayed in a merged inbox

## Security Note

The `credentials/` and `tokens/` directories contain sensitive authentication data. Keep these directories secure and don't commit them to version control. They should already be in your `.gitignore`.

## Troubleshooting

### Error: "Access blocked: [app name] can only be used within its organization" (Error 403: org_internal)

This error occurs when your OAuth app is configured as "Internal" in Google Cloud Console, which restricts access to users within the same Google Workspace organization. If you're trying to authenticate with a personal Gmail account, you'll need to change this setting.

**Solution:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to "APIs & Services" > "OAuth consent screen"
4. Change "User type" from "Internal" to "External"
5. Fill in the required app information (app name, support email, etc.)
6. Save and try authenticating again

**Alternative:** If you need to keep it as "Internal", you can add your personal Gmail address as a test user in the OAuth consent screen settings.

