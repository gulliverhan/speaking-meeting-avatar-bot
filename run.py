#!/usr/bin/env python3
"""Entry point for the Recall.ai ElevenLabs Meeting Bot server."""

import argparse
import os


def main():
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="Start the Recall.ai ElevenLabs Meeting Bot server"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "7014")),
        help="Port to listen on",
    )
    parser.add_argument(
        "--local-dev",
        action="store_true",
        help="Run in local development mode with ngrok support",
    )

    args = parser.parse_args()

    # Import here to avoid circular imports
    from app.main import start_server

    start_server(
        host=args.host,
        port=args.port,
        local_dev=args.local_dev,
    )


if __name__ == "__main__":
    main()
