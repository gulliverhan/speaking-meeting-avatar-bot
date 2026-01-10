"""Ngrok tunnel detection and URL management for local development."""

import os
from typing import Optional

import httpx

from utils.logger import logger

# Flag to indicate local development mode
LOCAL_DEV_MODE = os.path.exists(".local_dev_mode")

# Cache for ngrok URLs
_ngrok_urls: list[str] = []


def load_ngrok_urls() -> list[str]:
    """Load ngrok tunnel URLs from the local ngrok API.

    Returns:
        List of public ngrok URLs.
    """
    global _ngrok_urls

    try:
        # Query ngrok's local API for active tunnels
        response = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            tunnels = data.get("tunnels", [])
            _ngrok_urls = [
                t["public_url"]
                for t in tunnels
                if t.get("public_url", "").startswith("https://")
            ]
            logger.info(f"Found {len(_ngrok_urls)} ngrok tunnel(s)")
            return _ngrok_urls
    except Exception as e:
        logger.debug(f"Could not connect to ngrok API: {e}")

    return []


def get_ngrok_url() -> Optional[str]:
    """Get the first available ngrok HTTPS URL.

    Returns:
        The ngrok public URL, or None if not available.
    """
    if not _ngrok_urls:
        load_ngrok_urls()

    if _ngrok_urls:
        return _ngrok_urls[0]
    return None


def get_public_url() -> Optional[str]:
    """Get the public base URL for external access.
    
    Checks in order:
    1. BASE_URL environment variable
    2. ngrok URL (if in local dev mode)
    3. Returns None if no public URL available

    Returns:
        The public URL, or None if not available.
    """
    # Check for explicit BASE_URL first
    env_base_url = os.getenv("BASE_URL")
    if env_base_url:
        return env_base_url
    
    # In local dev mode, try ngrok
    if LOCAL_DEV_MODE:
        return get_ngrok_url()
    
    return None


def determine_websocket_url(
    base_url: Optional[str] = None,
    request=None,
) -> tuple[str, Optional[str]]:
    """Determine the WebSocket URL to use for callbacks.

    In local dev mode, prioritizes ngrok URLs.
    In production, uses BASE_URL environment variable or request host.

    Args:
        base_url: Optional explicit base URL to use.
        request: Optional FastAPI request for host detection.

    Returns:
        Tuple of (websocket_url, temp_client_id).
        temp_client_id is used for ngrok URL tracking in local dev.
    """
    temp_client_id = None

    # Check for explicit BASE_URL
    if base_url:
        return base_url, temp_client_id

    env_base_url = os.getenv("BASE_URL")
    if env_base_url:
        return env_base_url, temp_client_id

    # In local dev mode, try ngrok
    if LOCAL_DEV_MODE:
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            logger.info(f"Using ngrok URL: {ngrok_url}")
            return ngrok_url, temp_client_id

    # Fall back to request host
    if request:
        host = request.headers.get("host", "localhost:7014")
        scheme = "https" if "https" in str(request.url) else "http"
        return f"{scheme}://{host}", temp_client_id

    # Default fallback
    port = os.getenv("PORT", "7014")
    return f"http://localhost:{port}", temp_client_id


def log_ngrok_status() -> None:
    """Log the current ngrok tunnel status."""
    urls = load_ngrok_urls()
    if urls:
        logger.info(f"Active ngrok tunnels: {urls}")
    else:
        logger.info("No active ngrok tunnels found")


def update_ngrok_client_id(old_id: str, new_id: str) -> None:
    """Update client ID mapping for ngrok URL tracking.

    Args:
        old_id: The temporary client ID.
        new_id: The actual client ID.
    """
    # Placeholder for URL-to-client tracking if needed
    logger.debug(f"Updated client ID mapping: {old_id} -> {new_id}")


def release_ngrok_url(client_id: str) -> None:
    """Release an ngrok URL associated with a client.

    Args:
        client_id: The client ID to release.
    """
    logger.debug(f"Released ngrok URL for client: {client_id}")
