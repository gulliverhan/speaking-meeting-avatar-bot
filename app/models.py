"""Pydantic models for the meeting bot API."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class BotRequest(BaseModel):
    """Request model for creating a meeting bot."""

    meeting_url: str = Field(..., description="URL of the meeting to join")
    agent_name: str = Field(
        default="meeting_facilitator",
        description="Name of the agent to use (folder name in agents/)",
    )
    bot_name: Optional[str] = Field(
        default=None,
        description="Display name for the bot in the meeting",
    )
    bot_image: Optional[str] = Field(
        default=None,
        description="URL for the bot's avatar image",
    )
    entry_message: Optional[str] = Field(
        default=None,
        description="Message to send when joining the meeting",
    )
    audio_mode: str = Field(
        default="browser_sdk",
        description="Audio handling mode: 'browser_sdk' (ElevenLabs SDK in browser), 'server' (server-side processing), or 'audio_test' (simple audio test)",
    )
    speaker_context: bool = Field(
        default=False,
        description="Send speaker identification to ElevenLabs (e.g., '[John is speaking]')",
    )
    extra: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the bot",
    )


class BotResponse(BaseModel):
    """Response model for bot creation."""

    bot_id: str = Field(..., description="The created bot ID")
    agent_name: str = Field(..., description="The agent being used")


class BotRemoveRequest(BaseModel):
    """Request model for removing a bot."""

    bot_id: Optional[str] = Field(
        default=None,
        description="Bot ID to remove (optional if in URL)",
    )


class ToolCallRequest(BaseModel):
    """Request model for tool webhook calls from ElevenLabs."""

    tool_name: str = Field(..., description="Name of the tool being called")
    tool_call_id: str = Field(..., description="Unique ID for this tool call")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to the tool",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="The conversation/session ID",
    )


class ToolCallResponse(BaseModel):
    """Response model for tool webhook calls."""

    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Optional[Any] = Field(
        default=None,
        description="Result data from the tool",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the tool call failed",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="healthy")
    version: str = Field(default="0.1.0")
