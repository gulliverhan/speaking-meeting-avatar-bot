"""Main FastAPI application for the Recall.ai ElevenLabs Meeting Bot."""

import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.logger import configure_logger
from utils.ngrok import load_ngrok_urls

# Configure logging
logger = configure_logger()


async def api_key_middleware(request: Request, call_next):
    """Middleware to check for Recall.ai API key in headers.

    Args:
        request: The incoming FastAPI request.
        call_next: The next middleware or route handler.

    Returns:
        The response from the next handler, or a 401 error if API key
        is missing for protected routes.
    """
    # Skip API key check for docs, health, agents list, static files, and openapi endpoints
    if request.url.path in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/agents"]:
        return await call_next(request)

    # Skip for static files (needed for Recall.ai webpage rendering)
    if request.url.path.startswith("/static/"):
        return await call_next(request)

    # Skip for WebSocket upgrade requests (handled separately)
    if request.url.path.startswith("/ws/"):
        request.state.api_key = None
        return await call_next(request)

    # Skip for webhooks (ElevenLabs tools and Recall.ai transcription)
    if request.url.path.startswith("/webhooks/"):
        return await call_next(request)

    # Check header first, then fall back to environment variable
    api_key = request.headers.get("x-recall-api-key") or os.getenv("RECALL_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "message": "Missing Recall.ai API key in x-recall-api-key header or RECALL_API_KEY env var"
            },
        )

    request.state.api_key = api_key
    return await call_next(request)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI application instance.
    """
    app = FastAPI(
        title="Recall.ai ElevenLabs Meeting Bot",
        description="API for deploying AI-powered meeting bots with "
        "Recall.ai connectivity and ElevenLabs Conversational AI.",
        version="0.1.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
    )

    # Add API key middleware
    app.middleware("http")(api_key_middleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from app.routes import router as app_router
    from app.webhooks import router as webhooks_router
    from app.websockets import router as websocket_router

    app.include_router(app_router)
    app.include_router(webhooks_router)
    app.include_router(websocket_router)

    # Mount static files for the agent webpage
    import os
    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
        logger.info(f"Serving static files from {static_path}")
    
    # Serve participant frames for image analysis via Replicate
    frames_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "participant_frames")
    os.makedirs(frames_path, exist_ok=True)  # Ensure directory exists
    app.mount("/participant_frames", StaticFiles(directory=frames_path), name="participant_frames")
    logger.info(f"Serving participant frames from {frames_path}")

    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 7014,
    local_dev: bool = False,
):
    """Start the Uvicorn server for the FastAPI application.

    Args:
        host: The host address to bind to.
        port: The port to listen on.
        local_dev: Whether to run in local development mode with ngrok.
    """
    # Use PORT environment variable if set
    server_port = int(os.getenv("PORT", str(port)))
    server_host = os.getenv("HOST", host)

    logger.info(f"Starting server on {server_host}:{server_port}")

    if local_dev:
        print("\n⚠️  Starting in local development mode")
        ngrok_urls = load_ngrok_urls()

        if ngrok_urls:
            print(f"✅ {len(ngrok_urls)} ngrok tunnel(s) available")
            for i, url in enumerate(ngrok_urls):
                print(f"   Tunnel {i + 1}: {url}")
        else:
            print("⚠️  No ngrok URLs found. Start ngrok with: ngrok http 7014")
        print()

    # Build uvicorn command
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        server_host,
        "--port",
        str(server_port),
    ]

    if local_dev:
        args.append("--reload")
        with open(".local_dev_mode", "w") as f:
            f.write("true")
    else:
        if os.path.exists(".local_dev_mode"):
            os.remove(".local_dev_mode")

    os.execv(sys.executable, args)
