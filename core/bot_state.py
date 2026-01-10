"""Bot state management for tracking active bots."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BotState:
    """State for a single bot instance.

    Attributes:
        client_id: Unique identifier for this bot.
        agent_name: Name of the agent configuration being used.
        meeting_url: URL of the meeting the bot joined.
        bot_name: Display name of the bot in the meeting.
        recall_bot_id: Bot ID from Recall.ai API.
        elevenlabs_agent_id: Agent ID for ElevenLabs conversation.
        elevenlabs_conversation_id: Conversation ID from ElevenLabs.
        current_expression: Current avatar expression.
        wants_to_speak: Whether the hand-raised overlay is active.
        audio_mode: Audio handling mode ('browser_sdk' or 'server').
        speaker_context: Send speaker identification to ElevenLabs.
        extra: Additional metadata.
    """

    client_id: str
    agent_name: str
    meeting_url: str
    bot_name: str = "AI Assistant"
    recall_bot_id: Optional[str] = None
    elevenlabs_agent_id: Optional[str] = None
    elevenlabs_conversation_id: Optional[str] = None
    current_expression: str = "neutral"
    wants_to_speak: bool = False
    audio_mode: str = "browser_sdk"
    speaker_context: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# Global registry of active bots
# Key: client_id, Value: BotState
BOT_STATES: dict[str, BotState] = {}
