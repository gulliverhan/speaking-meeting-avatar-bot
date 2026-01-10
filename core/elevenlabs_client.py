"""ElevenLabs Conversational AI WebSocket client.

This module provides a client for connecting to ElevenLabs' Conversational AI
Agent platform via WebSocket. It handles bidirectional audio streaming between
the meeting audio and ElevenLabs' AI agent.

Adapted from speaking-meeting-bot project.
"""

import asyncio
import base64
import json
import os
from typing import Any, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol

from utils.logger import logger


class ElevenLabsClient:
    """Client for ElevenLabs Conversational AI WebSocket API.

    This client manages the WebSocket connection to ElevenLabs' Conversational
    AI platform, handling audio input/output and conversation state.

    Attributes:
        agent_id: The ElevenLabs agent ID to connect to.
        api_key: Optional API key for private agents.
        ws: The WebSocket connection instance.
        is_connected: Whether the client is currently connected.
        conversation_id: The current conversation session ID.
    """

    ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"

    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        on_audio_received: Optional[Callable[[bytes], Any]] = None,
        on_transcript: Optional[Callable[[str, str], Any]] = None,
        on_agent_response: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
    ):
        """Initialize the ElevenLabs client.

        Args:
            agent_id: The ElevenLabs agent ID to connect to.
            api_key: Optional API key for private agents.
            on_audio_received: Callback for when audio is received from the agent.
            on_transcript: Callback for transcription events (role, text).
            on_agent_response: Callback for agent text responses.
            on_tool_call: Callback for tool calls from the agent.
        """
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.on_audio_received = on_audio_received
        self.on_transcript = on_transcript
        self.on_agent_response = on_agent_response
        self.on_tool_call = on_tool_call

        self.ws: Optional[WebSocketClientProtocol] = None
        self.is_connected = False
        self.conversation_id: Optional[str] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._closing = False

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with agent ID.

        Returns:
            The full WebSocket URL for connecting to the agent.
        """
        return f"{self.ELEVENLABS_WS_URL}?agent_id={self.agent_id}"

    async def connect(self) -> bool:
        """Establish WebSocket connection to ElevenLabs.

        Returns:
            True if connection was successful, False otherwise.
        """
        if self.is_connected:
            logger.warning("Already connected to ElevenLabs")
            return True

        try:
            url = self._build_ws_url()
            headers = {}
            if self.api_key:
                headers["xi-api-key"] = self.api_key

            logger.info(f"Connecting to ElevenLabs agent: {self.agent_id}")
            self.ws = await websockets.connect(
                url,
                extra_headers=headers if headers else None,
                ping_interval=20,
                ping_timeout=10,
            )

            self.is_connected = True
            self._closing = False
            logger.info("Connected to ElevenLabs WebSocket")

            # Start the receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._closing = True

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self.ws = None

        self.is_connected = False
        self.conversation_id = None
        logger.info("Disconnected from ElevenLabs")

    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data to ElevenLabs.

        The audio should be 16-bit PCM at 16kHz sample rate.

        Args:
            audio_data: Raw PCM audio bytes to send.

        Returns:
            True if audio was sent successfully, False otherwise.
        """
        if not self.is_connected or not self.ws:
            logger.warning("Cannot send audio: not connected")
            return False

        try:
            # Encode audio as base64
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Create the audio input message
            message = {"user_audio_chunk": audio_base64}

            await self.ws.send(json.dumps(message))
            return True

        except Exception as e:
            logger.error(f"Error sending audio to ElevenLabs: {e}")
            return False

    async def send_contextual_update(self, context: str) -> bool:
        """Send a contextual update to the conversation.

        Args:
            context: The contextual information to add.

        Returns:
            True if the update was sent successfully, False otherwise.
        """
        if not self.is_connected or not self.ws:
            logger.warning("Cannot send context: not connected")
            return False

        try:
            message = {"type": "contextual_update", "text": context}
            await self.ws.send(json.dumps(message))
            logger.debug(f"Sent contextual update: {context[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error sending contextual update: {e}")
            return False

    async def _receive_loop(self) -> None:
        """Main loop for receiving messages from ElevenLabs."""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                if self._closing:
                    break
                await self._handle_message(message)

        except websockets.exceptions.ConnectionClosed as e:
            if not self._closing:
                logger.info(f"ElevenLabs connection closed: {e}")
        except Exception as e:
            if not self._closing:
                logger.error(f"Error in receive loop: {e}")
        finally:
            self.is_connected = False

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """Handle an incoming message from ElevenLabs.

        Args:
            raw_message: The raw message received from the WebSocket.
        """
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            data = json.loads(raw_message)
            event_type = data.get("type")

            if event_type == "conversation_initiation_metadata":
                self.conversation_id = data.get("conversation_id")
                logger.info(f"Conversation initiated: {self.conversation_id}")

            elif event_type == "audio":
                # Agent audio response
                audio_base64 = data.get("audio_event", {}).get("audio_base_64")
                if audio_base64 and self.on_audio_received:
                    audio_bytes = base64.b64decode(audio_base64)
                    await self._call_callback(self.on_audio_received, audio_bytes)

            elif event_type == "agent_response":
                # Agent text response
                text = data.get("agent_response_event", {}).get("agent_response")
                if text and self.on_agent_response:
                    await self._call_callback(self.on_agent_response, text)

            elif event_type == "user_transcript":
                # User speech transcription
                text = data.get("user_transcription_event", {}).get("user_transcript")
                if text and self.on_transcript:
                    await self._call_callback(self.on_transcript, "user", text)

            elif event_type == "tool_call":
                # Agent is calling a tool
                tool_name = data.get("tool_call_event", {}).get("tool_name")
                parameters = data.get("tool_call_event", {}).get("parameters", {})
                if tool_name and self.on_tool_call:
                    await self._call_callback(self.on_tool_call, tool_name, parameters)

            elif event_type == "agent_response_correction":
                logger.debug("Agent response correction received")

            elif event_type == "interruption":
                logger.debug("User interruption detected")

            elif event_type == "ping":
                # Respond to ping with pong
                pong = {
                    "type": "pong",
                    "event_id": data.get("ping_event", {}).get("event_id"),
                }
                if self.ws:
                    await self.ws.send(json.dumps(pong))

            elif event_type == "internal_tentative_agent_response":
                pass  # Tentative response before finalization

            else:
                logger.debug(f"Unhandled event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _call_callback(self, callback: Callable, *args: Any) -> None:
        """Call a callback, handling both sync and async functions.

        Args:
            callback: The callback function to call.
            *args: Arguments to pass to the callback.
        """
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in callback: {e}")


class ElevenLabsBridge:
    """Bridge between Recall.ai audio and ElevenLabs Agent.

    This class manages the audio flow between a meeting (via Recall.ai)
    and an ElevenLabs Conversational AI agent.

    Attributes:
        client_id: The unique client/session identifier.
        agent_id: The ElevenLabs agent ID.
        elevenlabs_client: The ElevenLabs WebSocket client.
        send_to_meeting: Callback to send audio back to the meeting.
    """

    def __init__(
        self,
        client_id: str,
        agent_id: str,
        api_key: Optional[str] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
    ):
        """Initialize the bridge.

        Args:
            client_id: Unique identifier for this session.
            agent_id: The ElevenLabs agent ID to connect to.
            api_key: Optional ElevenLabs API key.
            on_tool_call: Callback for tool calls from the agent.
        """
        self.client_id = client_id
        self.agent_id = agent_id
        self.send_to_meeting: Optional[Callable[[bytes], Any]] = None
        self.on_tool_call = on_tool_call

        self.elevenlabs_client = ElevenLabsClient(
            agent_id=agent_id,
            api_key=api_key,
            on_audio_received=self._on_audio_from_agent,
            on_transcript=self._on_transcript,
            on_agent_response=self._on_agent_response,
            on_tool_call=self._on_tool_call,
        )

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self.elevenlabs_client.conversation_id

    async def start(self) -> bool:
        """Start the bridge by connecting to ElevenLabs.

        Returns:
            True if connection was successful, False otherwise.
        """
        logger.info(f"Starting ElevenLabs bridge for client {self.client_id}")
        return await self.elevenlabs_client.connect()

    async def stop(self) -> None:
        """Stop the bridge and disconnect from ElevenLabs."""
        logger.info(f"Stopping ElevenLabs bridge for client {self.client_id}")
        await self.elevenlabs_client.disconnect()

    async def send_audio_to_agent(self, audio_data: bytes) -> bool:
        """Send meeting audio to the ElevenLabs agent.

        Args:
            audio_data: Raw PCM audio from the meeting.

        Returns:
            True if audio was sent successfully, False otherwise.
        """
        return await self.elevenlabs_client.send_audio(audio_data)

    def set_meeting_audio_callback(self, callback: Callable[[bytes], Any]) -> None:
        """Set the callback for sending audio to the meeting.

        Args:
            callback: Function to call with audio bytes for the meeting.
        """
        self.send_to_meeting = callback

    async def _on_audio_from_agent(self, audio_data: bytes) -> None:
        """Handle audio received from ElevenLabs agent.

        Args:
            audio_data: Raw PCM audio bytes from the agent.
        """
        if self.send_to_meeting:
            try:
                result = self.send_to_meeting(audio_data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error sending audio to meeting: {e}")

    def _on_transcript(self, role: str, text: str) -> None:
        """Handle transcription events.

        Args:
            role: 'user' or 'agent'.
            text: The transcribed text.
        """
        logger.info(f"[{role.upper()}] {text}")

    def _on_agent_response(self, text: str) -> None:
        """Handle agent text responses.

        Args:
            text: The agent's response text.
        """
        logger.debug(f"Agent response: {text[:100]}...")

    async def _on_tool_call(self, tool_name: str, parameters: dict) -> None:
        """Handle tool calls from the agent.

        Args:
            tool_name: Name of the tool being called.
            parameters: Tool parameters.
        """
        logger.info(f"Tool call: {tool_name}({parameters})")
        if self.on_tool_call:
            try:
                result = self.on_tool_call(tool_name, parameters)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in tool call handler: {e}")


# Store active bridges by client_id
ELEVENLABS_BRIDGES: dict[str, ElevenLabsBridge] = {}


async def create_bridge(
    client_id: str,
    agent_id: str,
    api_key: Optional[str] = None,
    on_tool_call: Optional[Callable[[str, dict], Any]] = None,
) -> ElevenLabsBridge:
    """Create and start a new ElevenLabs bridge.

    Args:
        client_id: Unique identifier for this session.
        agent_id: The ElevenLabs agent ID to connect to.
        api_key: Optional ElevenLabs API key.
        on_tool_call: Callback for tool calls.

    Returns:
        The created and connected bridge instance.

    Raises:
        ConnectionError: If the bridge fails to connect.
    """
    bridge = ElevenLabsBridge(
        client_id=client_id,
        agent_id=agent_id,
        api_key=api_key,
        on_tool_call=on_tool_call,
    )

    if not await bridge.start():
        raise ConnectionError(f"Failed to connect to ElevenLabs agent: {agent_id}")

    ELEVENLABS_BRIDGES[client_id] = bridge
    return bridge


async def get_bridge(client_id: str) -> Optional[ElevenLabsBridge]:
    """Get an existing bridge by client ID.

    Args:
        client_id: The client ID to look up.

    Returns:
        The bridge instance if found, None otherwise.
    """
    return ELEVENLABS_BRIDGES.get(client_id)


async def remove_bridge(client_id: str) -> None:
    """Stop and remove a bridge.

    Args:
        client_id: The client ID of the bridge to remove.
    """
    bridge = ELEVENLABS_BRIDGES.pop(client_id, None)
    if bridge:
        await bridge.stop()
