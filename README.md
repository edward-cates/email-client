# email-client
An email client to classify my emails where everything runs locally.

## Running it

```
pip install -r requirements.txt
make up # Run the application
make lint # Lint the code
```

## Gmail Auth

See [Gmail API Setup](src/gmail/README.md) for authentication setup instructions.

### Multi-Account Support

This email client supports multiple Gmail accounts (up to 9) in a single merged inbox. To add accounts:

1. Place credential files in `src/gmail/credentials/`:
   - `credentials-1.json` for account1
   - `credentials-2.json` for account2
   - ... up to `credentials-9.json` for account9

2. The application will automatically discover all credential files and display them in the sidebar.

3. Authenticate each account by clicking "Sign in" next to the account in the sidebar.

4. Once authenticated, emails from all accounts will be merged into a single inbox, sorted by date (newest first).

5. Each email shows which account it belongs to, and actions (archive, add labels) are performed on the correct account automatically.

## Custom Labels

All Gmail accounts should have a custom label called "marketing". Emails with this label will display a blue "Marketing" badge in the email list.

