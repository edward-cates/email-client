from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

app = FastAPI()

# Get the directory where this file is located
static_dir = Path(__file__).parent

# Mount static files if needed
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    """Serve the main Gmail client page"""
    html_file = static_dir / "index.html"
    return FileResponse(html_file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

