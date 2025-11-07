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

The email client supports custom labels that can be added to emails. Labels are configured in `src/gmail/labels.yaml`.

### Adding Labels

To add a new label, edit `src/gmail/labels.yaml`:

```yaml
labels:
  - name: marketing
    css_class: marketing
  - name: boring noti
    css_class: boring-noti
  - name: event
    css_class: event
  - name: newsletter
    css_class: newsletter
  - name: my new label
    css_class: my-new-label
```

Each label requires:
- `name`: The label name as it appears in Gmail (case-insensitive)
- `css_class`: A CSS class name (use kebab-case, e.g., `my-new-label`)

### Label Behavior

- Labels appear as badges next to the sender name in the email list
- Click a badge to add/remove the label from an email
- New labels automatically work with a default gray color
- To customize colors, add CSS in `src/web/index.html`:

```css
.label-badge.my-new-label {
    background-color: #your-color;
}
.label-badge.my-new-label:hover:not(.loading) {
    background-color: #your-darker-color;
}
```

### Default Labels

The default configuration includes:
- `marketing` (blue)
- `boring noti` (gray)
- `event` (red)
- `newsletter` (green)

After modifying `labels.yaml`, restart the server for changes to take effect.

