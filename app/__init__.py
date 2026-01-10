"""Main application package for the Recall.ai ElevenLabs Meeting Bot."""

from app.main import create_app

app = create_app()

__all__ = ["app", "create_app"]
