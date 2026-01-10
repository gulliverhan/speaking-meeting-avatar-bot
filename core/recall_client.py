"""Recall.ai API client for creating and managing meeting bots.

This module provides functions to interact with the Recall.ai API
for creating bots that join video meetings with audio/video streaming.

API Reference: https://docs.recall.ai/reference/bot_create
"""

import os
from typing import Any, Optional

import httpx

from utils.logger import logger

# Recall.ai API base URL
RECALL_API_BASE = "https://us-west-2.recall.ai/api/v1"


class RecallClient:
    """Client for interacting with the Recall.ai API.

    Attributes:
        api_key: The Recall.ai API key.
        base_url: The API base URL.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Recall.ai client.

        Args:
            api_key: Optional API key. If not provided, uses RECALL_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("RECALL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Recall.ai API key is required. Set RECALL_API_KEY environment variable."
            )
        self.base_url = RECALL_API_BASE

    def _get_headers(self) -> dict[str, str]:
        """Get the request headers with authentication.

        Returns:
            Dictionary of headers.
        """
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_bot(
        self,
        meeting_url: str,
        bot_name: str = "AI Assistant",
        bot_image: Optional[str] = None,
        output_media: Optional[dict[str, Any]] = None,
        input_media: Optional[dict[str, Any]] = None,
        audio_websocket_url: Optional[str] = None,
        enable_transcription: bool = False,
        transcription_webhook_url: Optional[str] = None,
        video_websocket_url: Optional[str] = None,
        variant: Optional[str] = None,
        **kwargs,
    ) -> Optional[dict[str, Any]]:
        """Create a bot to join a meeting.

        Args:
            meeting_url: URL of the meeting to join.
            bot_name: Display name for the bot.
            bot_image: Optional URL for bot avatar.
            output_media: Output media configuration (e.g., webpage rendering).
            input_media: Input media configuration (e.g., microphone routing to webpage).
            audio_websocket_url: WebSocket URL to receive meeting audio in real-time.
            enable_transcription: Whether to enable Recall.ai real-time transcription.
            transcription_webhook_url: Webhook URL for receiving transcription events.
            video_websocket_url: WebSocket URL to receive per-participant video frames.
            variant: Bot variant for performance (e.g., "4_core", "gpu").
            **kwargs: Additional bot configuration options.

        Returns:
            Bot data including ID if successful, None otherwise.
        """
        payload: dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
        }
        
        # Use more powerful variant for output media features
        # Can be set per-platform or globally with "name"
        if variant:
            # Try platform-specific format first for better compatibility
            payload["variant"] = {
                "zoom": variant,
                "google_meet": variant,
                "microsoft_teams": variant,
            }
            logger.info(f"Using bot variant: {variant}")

        # Configure output media (video feed)
        # This tells Recall.ai to render a webpage as the bot's camera
        if output_media:
            payload["output_media"] = output_media
        
        # Configure input media (microphone routing)
        # This routes meeting audio to the webpage's getUserMedia
        if input_media:
            payload["input_media"] = input_media
            logger.info("Configured input media for webpage microphone")

        # Configure recording settings
        recording_config: dict[str, Any] = {}
        
        # Configure real-time audio streaming via WebSocket
        # This is SEPARATE from output - prevents echo!
        if audio_websocket_url:
            recording_config["audio_mixed_raw"] = {
                "websocket_url": audio_websocket_url
            }
            logger.info(f"Configured audio input WebSocket: {audio_websocket_url}")

        # Enable real-time transcription with Recall.ai's built-in provider
        # See: https://docs.recall.ai/docs/bot-real-time-transcription
        if enable_transcription and transcription_webhook_url:
            logger.info(f"Enabling Recall.ai transcription with webhook: {transcription_webhook_url}")
            
            # Configure transcript provider (using Recall's built-in streaming)
            recording_config["transcript"] = {
                "provider": {
                    "recallai_streaming": {
                        "language_code": "auto",
                        "filter_profanity": False,
                        "mode": "prioritize_accuracy"
                    }
                },
                # Enable per-participant diarization for speaker identification
                "diarization": {
                    "use_separate_streams_when_available": True
                }
            }
            
            # Initialize realtime_endpoints list
            if "realtime_endpoints" not in recording_config:
                recording_config["realtime_endpoints"] = []
            
            # Add transcription webhook endpoint
            recording_config["realtime_endpoints"].append({
                "type": "webhook",
                "url": transcription_webhook_url,
                "events": ["transcript.data"]  # Only final results, not partials
            })
        
        # Enable separate video per participant - this gives us participant info!
        # See: https://docs.recall.ai/docs/how-to-get-separate-videos-per-participant-realtime
        # Required: gallery_view_v2 layout
        recording_config["video_mixed_layout"] = "gallery_view_v2"
        recording_config["video_separate_png"] = {}
        
        # Add WebSocket endpoint for video frames if URL provided
        if video_websocket_url:
            if "realtime_endpoints" not in recording_config:
                recording_config["realtime_endpoints"] = []
            
            recording_config["realtime_endpoints"].append({
                "type": "websocket",
                "url": video_websocket_url,
                "events": ["video_separate_png.data"]
            })
            logger.info(f"Configured video WebSocket: {video_websocket_url}")

        if recording_config:
            payload["recording_config"] = recording_config

        # Add bot image if provided
        if bot_image:
            payload["bot_image"] = bot_image

        # Add any additional options
        payload.update(kwargs)

        try:
            logger.info(f"Creating Recall.ai bot for: {meeting_url}")
            logger.debug(f"Bot payload: {payload}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/bot",
                    json=payload,
                    headers=self._get_headers(),
                    timeout=30.0,
                )

            if response.status_code == 201:
                data = response.json()
                bot_id = data.get("id")
                logger.info(f"Bot created successfully: {bot_id}")
                return data
            else:
                logger.error(
                    f"Failed to create bot: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error creating Recall.ai bot: {e}")
            return None

    async def get_bot(self, bot_id: str) -> Optional[dict[str, Any]]:
        """Get bot status and details.

        Args:
            bot_id: The bot ID.

        Returns:
            Bot data if found, None otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/bot/{bot_id}",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get bot: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return None

    async def get_meeting_participants(self, bot_id: str) -> list[dict[str, Any]]:
        """Get list of participants in the meeting.

        The bot data includes meeting_participants with info like:
        - id: Participant ID
        - name: Display name
        - is_host: Whether they're the host
        - platform: zoom, google_meet, etc.
        - events: join/leave times

        Args:
            bot_id: The bot ID.

        Returns:
            List of participant dictionaries.
        """
        try:
            bot_data = await self.get_bot(bot_id)
            if bot_data:
                participants = bot_data.get("meeting_participants", [])
                
                # Always log the raw meeting_participants data for debugging
                logger.info(f"Raw meeting_participants from API: {bot_data.get('meeting_participants')}")
                
                # Log all top-level keys to understand the response structure
                available_keys = list(bot_data.keys())
                logger.info(f"Bot data keys: {available_keys}")
                
                # Log specific potentially useful fields
                if "meeting_metadata" in bot_data:
                    logger.info(f"meeting_metadata: {bot_data['meeting_metadata']}")
                if "meeting_url" in bot_data:
                    logger.info(f"meeting_url: {bot_data['meeting_url']}")
                if "video_url" in bot_data:
                    logger.info(f"video_url present: {bool(bot_data['video_url'])}")
                    
                # Check for alternative participant data locations
                if "participants" in bot_data:
                    logger.info(f"Found 'participants' key: {bot_data['participants']}")
                    if not participants:
                        participants = bot_data.get("participants", [])
                
                logger.info(f"Returning {len(participants)} participants")
                return participants
            return []
        except Exception as e:
            logger.error(f"Error getting participants: {e}")
            return []

    async def get_video_screenshot(self, bot_id: str, participant_id: str = None) -> Optional[bytes]:
        """Capture a screenshot from the meeting video.

        Note: This uses the recording video API to get a frame.
        May require specific recording configuration.

        Args:
            bot_id: The bot ID.
            participant_id: Optional specific participant to capture.

        Returns:
            Image bytes if successful, None otherwise.
        """
        try:
            # First, check if recording is available
            bot_data = await self.get_bot(bot_id)
            if not bot_data:
                return None

            # Try to get video data endpoint
            # Note: This may require video recording to be enabled
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/bot/{bot_id}/video",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

            if response.status_code == 200:
                return response.content
            else:
                logger.warning(f"Video capture not available: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error capturing video: {e}")
            return None

    async def delete_bot(self, bot_id: str) -> bool:
        """Remove a bot from a meeting by making it leave the call.

        Note: Recall.ai doesn't allow deleting bots that have joined a call.
        Instead, we use the 'leave_call' endpoint to make the bot leave.

        Args:
            bot_id: The bot ID to remove.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Making Recall.ai bot leave call: {bot_id}")

            async with httpx.AsyncClient() as client:
                # Use leave_call endpoint for bots that have joined
                response = await client.post(
                    f"{self.base_url}/bot/{bot_id}/leave_call",
                    headers=self._get_headers(),
                    timeout=30.0,
                )

            if response.status_code in (200, 201, 204):
                logger.info(f"Bot {bot_id} left call successfully")
                return True
            else:
                logger.error(
                    f"Failed to make bot leave: {response.status_code} - {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Error removing bot from call: {e}")
            return False

    async def send_audio(self, bot_id: str, audio_base64: str) -> bool:
        """Send audio data to a bot's output via Output Audio endpoint.

        The audio should be MP3 format encoded as base64.

        Args:
            bot_id: The bot ID.
            audio_base64: Base64 encoded MP3 audio.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/bot/{bot_id}/output_audio",
                    json={"audio_base64": audio_base64},
                    headers=self._get_headers(),
                    timeout=10.0,
                )

            if response.status_code not in (200, 201, 204):
                logger.warning(f"Output audio response: {response.status_code} - {response.text[:200]}")
                
            return response.status_code in (200, 201, 204)

        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            return False
    
    async def start_output_audio(self, bot_id: str) -> bool:
        """Start the output audio stream for a bot.

        This must be called before sending audio chunks.

        Args:
            bot_id: The bot ID.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/bot/{bot_id}/output_audio",
                    json={"kind": "mp3"},
                    headers=self._get_headers(),
                    timeout=10.0,
                )

            return response.status_code in (200, 201, 204)

        except Exception as e:
            logger.error(f"Error starting output audio: {e}")
            return False
    
    async def stop_output_audio(self, bot_id: str) -> bool:
        """Stop the output audio stream for a bot.

        Args:
            bot_id: The bot ID.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/bot/{bot_id}/output_audio",
                    headers=self._get_headers(),
                    timeout=10.0,
                )

            return response.status_code in (200, 201, 204)

        except Exception as e:
            logger.error(f"Error stopping output audio: {e}")
            return False

    async def send_image(
        self,
        bot_id: str,
        image_data: bytes,
        content_type: str = "image/png",
    ) -> bool:
        """Send an image to the bot's video output.

        Args:
            bot_id: The bot ID.
            image_data: Image bytes to send.
            content_type: MIME type of the image.

        Returns:
            True if successful, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/bot/{bot_id}/output_video",
                    content=image_data,
                    headers={
                        **self._get_headers(),
                        "Content-Type": content_type,
                    },
                    timeout=10.0,
                )

            return response.status_code in (200, 201, 204)

        except Exception as e:
            logger.error(f"Error sending image: {e}")
            return False


# Convenience function for creating a client
def get_recall_client(api_key: Optional[str] = None) -> RecallClient:
    """Get a Recall.ai client instance.

    Args:
        api_key: Optional API key.

    Returns:
        RecallClient instance.
    """
    return RecallClient(api_key=api_key)
