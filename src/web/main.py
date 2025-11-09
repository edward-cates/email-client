import json
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gmail.auth import (  # noqa: E402
    complete_authorization,
    get_account_email,
    get_authorization_url,
    is_authenticated,
)
from gmail.config import discover_accounts  # noqa: E402
from gmail.service import (  # noqa: E402
    add_labels,
    archive_message,
    get_emails,
    get_label_name_mapping,
    get_message,
    get_message_count,
    parse_message_full,
    remove_labels,
)

app = FastAPI()


class LabelRequest(BaseModel):
    label_names: list[str]


class BulkArchiveRequest(BaseModel):
    message_ids: list[str]
    account_ids: dict[str, str]  # Map of message_id to account_id

# CORS middleware for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this file is located
static_dir = Path(__file__).parent
gmail_dir = Path(__file__).parent.parent / "gmail"


def load_labels_config():
    """Load labels configuration from YAML file"""
    labels_file = gmail_dir / "labels.yaml"

    with open(labels_file, encoding='utf-8') as f:
        return yaml.safe_load(f)


@app.get("/")
async def read_root():
    """Serve the main Gmail client page"""
    html_file = static_dir / "index.html"
    return FileResponse(html_file)


@app.get("/api/labels")
async def get_labels():
    """Get labels configuration"""
    return load_labels_config()


@app.get("/api/auth/status")
async def auth_status(account_id: str = Query("account1")):
    """Check authentication status for an account"""
    authenticated = is_authenticated(account_id)
    email = None
    if authenticated:
        email = get_account_email(account_id)
    return {
        "authenticated": authenticated,
        "account_id": account_id,
        "email": email
    }


@app.get("/api/auth/start")
async def auth_start(account_id: str = Query("account1")):
    """Start OAuth flow and return authorization URL"""
    try:
        auth_url, state = get_authorization_url(account_id)
        return {
            "authorization_url": auth_url,
            "state": state,
            "account_id": account_id
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting auth: {str(e)}") from e


@app.get("/auth/callback")
async def auth_callback(code: str, state: str):
    """OAuth callback endpoint"""
    try:
        complete_authorization(code, state)
        return RedirectResponse(url="/?auth_success=true")
    except Exception as e:
        return RedirectResponse(url=f"/?auth_error={str(e)}")


@app.get("/api/emails")
async def get_emails_endpoint(
    account_id: str = Query("account1"),
    max_results: int = Query(50, ge=1, le=500),
    page_token: str = Query(None)
):
    """Get emails for an authenticated account"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        emails, next_page_token = get_emails(account_id, max_results=max_results, page_token=page_token)
        return {
            "account_id": account_id,
            "emails": emails,
            "count": len(emails),
            "next_page_token": next_page_token
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching emails: {str(e)}") from e


@app.post("/api/emails/{message_id}/archive")
async def archive_email_endpoint(
    message_id: str,
    account_id: str = Query("account1")
):
    """Archive an email"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        archive_message(account_id, message_id)
        return {"success": True, "message_id": message_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error archiving email: {str(e)}") from e


@app.post("/api/emails/bulk-archive")
async def bulk_archive_endpoint(request: BulkArchiveRequest):
    """Archive multiple emails"""
    results = {"success": [], "failed": []}

    for message_id in request.message_ids:
        account_id = request.account_ids.get(message_id)
        if not account_id:
            results["failed"].append({"message_id": message_id, "error": "Account ID not found"})
            continue

        if not is_authenticated(account_id):
            results["failed"].append({"message_id": message_id, "error": "Account not authenticated"})
            continue

        try:
            archive_message(account_id, message_id)
            results["success"].append(message_id)
        except Exception as e:
            results["failed"].append({"message_id": message_id, "error": str(e)})

    return {
        "success": True,
        "archived_count": len(results["success"]),
        "failed_count": len(results["failed"]),
        "results": results
    }


@app.post("/api/emails/{message_id}/labels")
async def add_labels_endpoint(
    message_id: str,
    request: LabelRequest,
    account_id: str = Query("account1")
):
    """Add labels to an email"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        result = add_labels(account_id, message_id, request.label_names)
        return {"success": True, "message_id": message_id, "labels": result.get('labelIds', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding labels: {str(e)}") from e


@app.delete("/api/emails/{message_id}/labels")
async def remove_labels_endpoint(
    message_id: str,
    request: LabelRequest,
    account_id: str = Query("account1")
):
    """Remove labels from an email"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        result = remove_labels(account_id, message_id, request.label_names)
        return {"success": True, "message_id": message_id, "labels": result.get('labelIds', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing labels: {str(e)}") from e


@app.get("/api/accounts")
async def list_accounts():
    """List all discovered accounts and their authentication status"""
    accounts = []
    discovered = discover_accounts()

    for account_id in discovered:
        authenticated = is_authenticated(account_id)
        email = None
        if authenticated:
            email = get_account_email(account_id)
        accounts.append({
            "account_id": account_id,
            "email": email,
            "authenticated": authenticated
        })

    return {"accounts": accounts}


@app.get("/api/emails/merged")
async def get_merged_emails(
    max_results: int = Query(20, ge=1, le=100),
    page_token: str = Query(None)
):
    """Get merged emails from all authenticated accounts with pagination"""
    discovered = discover_accounts()
    all_emails = []
    account_page_tokens = {}

    # Parse page_token if provided (format: account1:token1,account2:token2)
    if page_token:
        for token_pair in page_token.split(','):
            if ':' in token_pair:
                acc_id, token = token_pair.split(':', 1)
                account_page_tokens[acc_id] = token

    # Fetch emails from each account
    next_tokens = {}
    for account_id in discovered:
        if is_authenticated(account_id):
            try:
                acc_page_token = account_page_tokens.get(account_id)
                emails, next_token = get_emails(account_id, max_results=max_results, page_token=acc_page_token)
                # Add account_id to each email so we know which account it's from
                for email in emails:
                    email['account_id'] = account_id
                all_emails.extend(emails)
                if next_token:
                    next_tokens[account_id] = next_token
            except Exception as e:
                # Log error but continue with other accounts
                print(f"Error fetching emails for {account_id}: {e}")

    # Sort by internalDate (newest first)
    all_emails.sort(key=lambda x: int(x.get('internalDate', 0) or 0), reverse=True)

    # Limit to max_results total
    all_emails = all_emails[:max_results]

    # Build next_page_token from accounts that have more pages
    next_page_token = None
    if next_tokens:
        next_page_token = ','.join([f"{acc_id}:{token}" for acc_id, token in next_tokens.items()])

    return {
        "emails": all_emails,
        "count": len(all_emails),
        "accounts_checked": len(discovered),
        "next_page_token": next_page_token
    }


@app.get("/api/emails/merged/stream")
async def get_merged_emails_stream(
    max_results: int = Query(20, ge=1, le=100),
    page_token: str = Query(None)
):
    """Stream merged emails from all authenticated accounts with progress updates"""
    def generate():
        discovered = discover_accounts()
        authenticated_accounts = [acc_id for acc_id in discovered if is_authenticated(acc_id)]

        if not authenticated_accounts:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No authenticated accounts'})}\n\n"
            return

        account_page_tokens = {}
        if page_token:
            for token_pair in page_token.split(','):
                if ':' in token_pair:
                    acc_id, token = token_pair.split(':', 1)
                    account_page_tokens[acc_id] = token

        all_emails = []
        next_tokens = {}
        total_accounts = len(authenticated_accounts)

        # Track progress across all accounts
        account_progress = {acc_id: {'current': 0, 'total': 0} for acc_id in authenticated_accounts}
        progress_queue = Queue()

        def account_progress_cb(current: int, total: int, acc_id: str):
            account_progress[acc_id] = {'current': current, 'total': total}
            # Calculate per-account progress percent
            account_percent = int(current / total * 100) if total > 0 else 0
            progress_data = {
                'type': 'progress',
                'account_id': acc_id,
                'current': current,
                'total': total,
                'progress_percent': account_percent
            }
            progress_queue.put(progress_data)

        # Fetch emails from each account
        for idx, account_id in enumerate(authenticated_accounts):
            try:
                yield f"data: {json.dumps({'type': 'account_start', 'account_id': account_id, 'account_num': idx + 1, 'total_accounts': total_accounts})}\n\n"

                acc_page_token = account_page_tokens.get(account_id)

                # Fetch emails in a thread to allow progress streaming
                emails_result = [None]
                exception_result = [None]

                def fetch_emails(
                    captured_account_id=account_id,
                    captured_acc_page_token=acc_page_token,
                    captured_emails_result=emails_result,
                    captured_exception_result=exception_result
                ):
                    try:
                        emails, next_token = get_emails(
                            captured_account_id,
                            max_results=max_results,
                            page_token=captured_acc_page_token,
                            progress_callback=account_progress_cb
                        )
                        captured_emails_result[0] = (emails, next_token)
                    except Exception as e:
                        captured_exception_result[0] = e

                fetch_thread = Thread(target=fetch_emails)
                fetch_thread.start()

                # Stream progress updates while fetching
                while fetch_thread.is_alive():
                    try:
                        # Check for progress updates (non-blocking)
                        while not progress_queue.empty():
                            progress_data = progress_queue.get_nowait()
                            yield f"data: {json.dumps(progress_data)}\n\n"
                    except Exception:
                        pass
                    time.sleep(0.1)  # Small delay to avoid busy waiting

                # Get any remaining progress updates
                while not progress_queue.empty():
                    progress_data = progress_queue.get_nowait()
                    yield f"data: {json.dumps(progress_data)}\n\n"

                fetch_thread.join()

                if exception_result[0]:
                    raise exception_result[0]

                result = emails_result[0]
                if result is None:
                    continue

                # Type narrowing: result is not None here
                emails, next_token = result  # type: ignore[assignment]

                # Add account_id to each email and stream them
                for email in emails:
                    email['account_id'] = account_id
                    all_emails.append(email)
                    yield f"data: {json.dumps({'type': 'email', 'email': email})}\n\n"

                if next_token:
                    next_tokens[account_id] = next_token

                yield f"data: {json.dumps({'type': 'account_complete', 'account_id': account_id, 'count': len(emails)})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'account_id': account_id, 'message': str(e)})}\n\n"

        # Sort by internalDate (newest first)
        all_emails.sort(key=lambda x: int(x.get('internalDate', 0) or 0), reverse=True)

        # Limit to max_results total
        all_emails = all_emails[:max_results]

        # Build next_page_token
        next_page_token = None
        if next_tokens:
            next_page_token = ','.join([f"{acc_id}:{token}" for acc_id, token in next_tokens.items()])

        # Send final result
        yield f"data: {json.dumps({'type': 'complete', 'count': len(all_emails), 'next_page_token': next_page_token})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/emails/total-count")
async def get_total_email_count():
    """Get total count of emails across all authenticated accounts"""
    discovered = discover_accounts()
    total_count = 0

    for account_id in discovered:
        if is_authenticated(account_id):
            try:
                count = get_message_count(account_id)
                total_count += count
            except Exception as e:
                print(f"Error getting count for {account_id}: {e}")

    return {
        "total_count": total_count,
        "accounts_checked": len(discovered)
    }


@app.get("/api/emails/account-counts")
async def get_account_email_counts():
    """Get email counts for each authenticated account"""
    discovered = discover_accounts()
    account_counts = {}

    for account_id in discovered:
        if is_authenticated(account_id):
            try:
                count = get_message_count(account_id)
                account_counts[account_id] = count
            except Exception as e:
                print(f"Error getting count for {account_id}: {e}")
                account_counts[account_id] = 0

    return {
        "account_counts": account_counts
    }


@app.get("/api/emails/stream")
async def get_emails_stream(
    account_id: str = Query("account1"),
    max_results: int = Query(50, ge=1, le=500),
    page_token: str = Query(None)
):
    """Stream emails for a single account with progress updates"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    def generate():
        progress_queue = Queue()

        def progress_cb(current: int, total: int, acc_id: str):
            progress_data = {
                'type': 'progress',
                'account_id': acc_id,
                'current': current,
                'total': total,
                'progress_percent': int(current / total * 100) if total > 0 else 0
            }
            progress_queue.put(progress_data)

        # Fetch emails in a thread to allow progress streaming
        emails_result: list[tuple[list[dict], str | None] | None] = [None]
        next_token_result: list[str | None] = [None]
        exception_result: list[Exception | None] = [None]

        def fetch_emails():
            try:
                emails, next_token = get_emails(
                    account_id,
                    max_results=max_results,
                    page_token=page_token,
                    progress_callback=progress_cb
                )
                emails_result[0] = (emails, next_token)
                next_token_result[0] = next_token
            except Exception as e:
                exception_result[0] = e

        fetch_thread = Thread(target=fetch_emails)
        fetch_thread.start()

        # Stream progress updates while fetching
        while fetch_thread.is_alive():
            try:
                while not progress_queue.empty():
                    progress_data = progress_queue.get_nowait()
                    yield f"data: {json.dumps(progress_data)}\n\n"
            except Exception:
                pass
            time.sleep(0.1)

        # Get any remaining progress updates
        while not progress_queue.empty():
            progress_data = progress_queue.get_nowait()
            yield f"data: {json.dumps(progress_data)}\n\n"

        fetch_thread.join()

        if exception_result[0]:
            yield f"data: {json.dumps({'type': 'error', 'account_id': account_id, 'message': str(exception_result[0])})}\n\n"
            return

        result = emails_result[0]
        if result is None:
            emails = []
            next_token = None
        else:
            emails, next_token = result

        # Stream each email
        for email in emails:
            email['account_id'] = account_id
            yield f"data: {json.dumps({'type': 'email', 'email': email})}\n\n"

        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'account_id': account_id, 'count': len(emails), 'next_page_token': next_token})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/emails/{message_id}")
async def get_email_details(
    message_id: str,
    account_id: str = Query("account1")
):
    """Get full email details including HTML body"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        message = get_message(account_id, message_id, format='full')
        if not message:
            raise HTTPException(status_code=404, detail="Email not found")

        label_id_to_name = get_label_name_mapping(account_id)
        parsed = parse_message_full(message, label_id_to_name)
        parsed['account_id'] = account_id
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching email: {str(e)}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
