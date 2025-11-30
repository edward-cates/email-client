import json
import os
import re
import socket
import sys
import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Thread

import httpx
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

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

# Import inference function (lazy import to avoid loading model at startup)
def _get_label_scores(email: dict) -> dict[str, int] | None:
    """Get label scores for an email using the model"""
    try:
        from classification.inference import predict_email_labels  # noqa: E402
        return predict_email_labels(email)
    except Exception:
        # If model is not available or there's an error, return None
        return None

app = FastAPI()


class LabelRequest(BaseModel):
    label_names: list[str]


class BulkArchiveRequest(BaseModel):
    message_ids: list[str]
    account_ids: dict[str, str]  # Map of message_id to account_id


class SkyvernRequest(BaseModel):
    email_html: str


class SummarizeRequest(BaseModel):
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
    """Get labels configuration with include_in_ml flag"""
    labels_config = load_labels_config()
    # Ensure each label has include_in_ml flag (defaults to True if not specified)
    if "labels" in labels_config:
        for label in labels_config["labels"]:
            if "include_in_ml" not in label:
                label["include_in_ml"] = True
    return labels_config


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
    results: dict[str, list] = {"success": [], "failed": []}

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
                # Also add model scores for emails without custom labels
                for email in emails:
                    email['account_id'] = account_id
                    # Add label scores if model is available
                    scores = _get_label_scores(email)
                    if scores:
                        email['label_scores'] = scores
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
                    # Add label scores if model is available
                    scores = _get_label_scores(email)
                    if scores:
                        email['label_scores'] = scores
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
            # Add label scores if model is available
            scores = _get_label_scores(email)
            if scores:
                email['label_scores'] = scores
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
        # Add label scores if model is available
        scores = _get_label_scores(parsed)
        if scores:
            parsed['label_scores'] = scores
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching email: {str(e)}") from e


@app.get("/api/ollama/available")
async def check_ollama_available():
    """Check if Ollama is available (port 11434 is listening), OLLAMA_SUMMARY_MODEL is set, and model exists locally"""
    try:
        # Check 1: Is Ollama running?
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        ollama_running = result == 0
        
        if not ollama_running:
            print("Ollama check: no (port 11434 not listening)")
            return {
                "available": False,
                "ollama_running": False,
                "model_configured": False,
                "model_exists": False,
                "model_name": None
            }
        
        # Check 2: Is OLLAMA_SUMMARY_MODEL set?
        summary_model = os.getenv('OLLAMA_SUMMARY_MODEL')
        has_model_config = bool(summary_model)
        
        if not has_model_config:
            print("Ollama check: no (OLLAMA_SUMMARY_MODEL not set)")
            return {
                "available": False,
                "ollama_running": True,
                "model_configured": False,
                "model_exists": False,
                "model_name": None
            }
        
        # Check 3: Does the model exist locally?
        model_exists = False
        available_models = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get('http://localhost:11434/api/tags')
                if response.status_code == 200:
                    data = response.json()
                    available_models = [m['name'] for m in data.get('models', [])]
                    # Check if the configured model exists (exact match or with :latest tag)
                    model_exists = (
                        summary_model in available_models or
                        f"{summary_model}:latest" in available_models or
                        any(summary_model == m.split(':')[0] for m in available_models)
                    )
        except Exception as e:
            print(f"Ollama check: error querying models - {e}")
        
        if not model_exists:
            print(f"Ollama check: WARNING - Model '{summary_model}' not found locally. Available models: {available_models}")
            print(f"Ollama check: Install with: ollama pull {summary_model}")
        else:
            print(f"Ollama check: yes (model '{summary_model}' found locally)")
        
        available = ollama_running and has_model_config and model_exists
        
        return {
            "available": available,
            "ollama_running": ollama_running,
            "model_configured": has_model_config,
            "model_exists": model_exists,
            "model_name": summary_model if has_model_config else None,
            "available_models": available_models if not model_exists else None  # Only return if model not found, for debugging
        }
    except Exception as e:
        print(f"Ollama check error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "available": False,
            "ollama_running": False,
            "model_configured": False,
            "model_exists": False,
            "model_name": None
        }


@app.get("/api/skyvern/available")
async def check_skyvern_available():
    """Check if Skyvern is available (API key is configured)"""
    api_key = os.getenv("SKYVERN_API_KEY")
    return {"available": bool(api_key)}


@app.post("/api/ollama/summarize")
async def summarize_emails(request: SummarizeRequest):
    """Summarize emails using Ollama with streaming"""
    print(f"[Ollama Summarize] Received request for {len(request.message_ids)} emails")
    
    # Check if Ollama is available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', 11434))
    sock.close()
    if result != 0:
        print("[Ollama Summarize] ERROR: Ollama is not available (port 11434 not listening)")
        raise HTTPException(status_code=503, detail="Ollama is not available")
    
    async def generate():
        try:
            # Fetch email details
            print(f"[Ollama Summarize] Fetching {len(request.message_ids)} email details...")
            email_texts = []
            for idx, message_id in enumerate(request.message_ids):
                account_id = request.account_ids.get(message_id)
                if not account_id:
                    print(f"[Ollama Summarize] WARNING: No account_id for message {message_id}")
                    continue
                if not is_authenticated(account_id):
                    print(f"[Ollama Summarize] WARNING: Account {account_id} not authenticated for message {message_id}")
                    continue
                
                try:
                    message = get_message(account_id, message_id, format='full')
                    if message:
                        label_id_to_name = get_label_name_mapping(account_id)
                        parsed = parse_message_full(message, label_id_to_name)
                        # Format email for summarization
                        body = parsed.get('text', '')
                        if not body and parsed.get('html'):
                            # Fallback to HTML if no text, but limit length
                            html = parsed.get('html', '')
                            # Simple HTML tag removal
                            body = re.sub(r'<[^>]+>', '', html)
                        # Limit body length to avoid token limits
                        body = body[:2000] if body else ''
                        email_text = f"From: {parsed.get('from', 'Unknown')}\n"
                        email_text += f"Subject: {parsed.get('subject', 'No subject')}\n"
                        email_text += f"Date: {parsed.get('date', 'Unknown')}\n"
                        email_text += f"Body: {body}\n"
                        email_texts.append(email_text)
                        print(f"[Ollama Summarize] Fetched email {idx+1}/{len(request.message_ids)}: {parsed.get('subject', 'No subject')[:50]}")
                except Exception as e:
                    print(f"[Ollama Summarize] ERROR fetching email {message_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not email_texts:
                print("[Ollama Summarize] ERROR: No emails to summarize")
                yield f"data: {json.dumps({'type': 'error', 'message': 'No emails to summarize'})}\n\n"
                return
            
            print(f"[Ollama Summarize] Prepared {len(email_texts)} emails for summarization")
            
            # Prepare prompt for Ollama
            emails_content = "\n\n---\n\n".join(email_texts)
            prompt = f"""Summarize the following emails into a concise markdown-formatted bullet list with emojis.

Requirements:
- Output ONLY the summary in markdown format (use - for bullets, ** for bold, etc.)
- Each bullet point should represent a key topic or theme from the emails
- Use appropriate emojis to make it visually appealing
- Do NOT include any meta-commentary like "I've summarized" or "Here are X bullets"
- Do NOT include any introductory text - start directly with the bullet list
- Output should be valid markdown that can be rendered as HTML

Emails:
{emails_content}

Summary (markdown format, bullets only, no meta-commentary):"""
            
            prompt_length = len(prompt)
            print(f"[Ollama Summarize] Prompt length: {prompt_length} characters")
            
            # Use the configured summary model
            model = os.getenv('OLLAMA_SUMMARY_MODEL')
            if not model:
                print("[Ollama Summarize] ERROR: OLLAMA_SUMMARY_MODEL not set")
                yield f"data: {json.dumps({'type': 'error', 'message': 'OLLAMA_SUMMARY_MODEL environment variable not configured'})}\n\n"
                return
            
            print(f"[Ollama Summarize] Using model: {model} (from OLLAMA_SUMMARY_MODEL env var)")
            
            # Now stream from Ollama
            print(f"[Ollama Summarize] Sending request to Ollama API with model '{model}'...")
            async with httpx.AsyncClient(timeout=300.0) as client:
                request_payload = {
                    'model': model,
                    'prompt': prompt,
                    'stream': True
                }
                print(f"[Ollama Summarize] Request payload keys: {list(request_payload.keys())}, model: {model}, prompt length: {len(prompt)}")
                
                async with client.stream(
                    'POST',
                    'http://localhost:11434/api/generate',
                    json=request_payload
                ) as response:
                    print(f"[Ollama Summarize] Response status: {response.status_code}")
                    
                    if response.status_code == 404:
                        error_msg = f'Model "{model}" not found. Please install it with: ollama pull {model}'
                        print(f"[Ollama Summarize] ERROR: {error_msg}")
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                        return
                    elif response.status_code == 400:
                        # Try to read error response
                        try:
                            error_body = await response.aread()
                            error_text = error_body.decode('utf-8', errors='ignore') if error_body else 'Bad request'
                            print(f"[Ollama Summarize] ERROR 400 - Response body: {error_text[:500]}")
                            error_msg = f'Ollama API error: {error_text[:200]}'
                        except Exception as e:
                            print(f"[Ollama Summarize] ERROR 400 - Could not read response: {e}")
                            error_msg = f'Ollama API error: Bad request (HTTP 400). Check model name and prompt format.'
                        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                        return
                    
                    response.raise_for_status()
                    print("[Ollama Summarize] Streaming response...")
                    buffer = ""
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    chunk = data['response']
                                    buffer += chunk
                                    chunk_count += 1
                                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                                if data.get('done', False):
                                    print(f"[Ollama Summarize] Complete. Received {chunk_count} chunks, total length: {len(buffer)}")
                                    break
                            except json.JSONDecodeError as e:
                                print(f"[Ollama Summarize] WARNING: JSON decode error on line: {line[:100]}")
                                continue
                    
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except httpx.HTTPStatusError as e:
            # Handle specific error codes
            print(f"[Ollama Summarize] HTTPStatusError: {e.response.status_code}")
            import traceback
            traceback.print_exc()
            
            if e.response.status_code == 404:
                error_msg = 'Model not found. Please install a model with: ollama pull llama3.2'
            elif e.response.status_code == 400:
                error_msg = 'Bad request. Check model name and request format.'
            elif e.response.status_code == 500:
                error_msg = 'Ollama server error. Check if the model is properly installed.'
            else:
                error_msg = f'Ollama API error: HTTP {e.response.status_code}'
            print(f"[Ollama Summarize] ERROR: {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        except Exception as e:
            print(f"[Ollama Summarize] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/skyvern/run-workflow")
async def run_skyvern_workflow(request: SkyvernRequest):
    """Run Skyvern workflow with email HTML"""
    api_key = os.getenv("SKYVERN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="SKYVERN_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.skyvern.com/v1/run/workflows",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key
                },
                json={
                    "workflow_id": "wpid_466366219251350078",
                    "parameters": {
                        "email_html": request.email_html
                    },
                    "proxy_location": "RESIDENTIAL",
                    "browser_session_id": None,
                    "browser_address": None,
                    "run_with": "agent",
                    "ai_fallback": False,
                    "extra_http_headers": {},
                    "cache_key": str(uuid.uuid4())
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Skyvern API error: {e.response.text}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Skyvern API: {str(e)}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
