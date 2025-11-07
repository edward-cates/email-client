import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gmail.auth import (
    complete_authorization,
    get_account_email,
    get_authorization_url,
    is_authenticated,
)
from gmail.service import get_emails

app = FastAPI()

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


@app.get("/")
async def read_root():
    """Serve the main Gmail client page"""
    html_file = static_dir / "index.html"
    return FileResponse(html_file)


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
        # Redirect to main page
        return RedirectResponse(url="/?auth_success=true")
    except Exception as e:
        return RedirectResponse(url=f"/?auth_error={str(e)}")


@app.get("/api/emails")
async def get_emails_endpoint(
    account_id: str = Query("account1"),
    max_results: int = Query(50, ge=1, le=500)
):
    """Get emails for an authenticated account"""
    if not is_authenticated(account_id):
        raise HTTPException(status_code=401, detail="Account not authenticated")

    try:
        emails = get_emails(account_id, max_results=max_results)
        return {
            "account_id": account_id,
            "emails": emails,
            "count": len(emails)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching emails: {str(e)}") from e


@app.get("/api/accounts")
async def list_accounts():
    """List all authenticated accounts"""
    accounts = []
    for account_id in ["account1", "account2"]:
        if is_authenticated(account_id):
            email = get_account_email(account_id)
            accounts.append({
                "account_id": account_id,
                "email": email,
                "authenticated": True
            })
        else:
            accounts.append({
                "account_id": account_id,
                "email": None,
                "authenticated": False
            })
    return {"accounts": accounts}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
