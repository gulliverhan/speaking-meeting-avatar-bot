"""ElevenLabs Agent Sync - Create and update conversational AI agents.

This module manages the lifecycle of ElevenLabs agents:
- Creates new agents from local config files
- Updates existing agents when config changes
- Tracks mapping between local agent names and ElevenLabs agent IDs
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from core.agent_manager import get_agent_config, list_agents
from utils.logger import logger
from utils.prompts import load_prompt

# File to store agent mappings
AGENT_MAPPING_FILE = Path(__file__).parent.parent / ".elevenlabs_agents.json"

# ElevenLabs API base URL
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Prefix for our agents to identify them
AGENT_PREFIX = "recall_"


def _get_api_key() -> str:
    """Get the ElevenLabs API key from environment."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable is required")
    return api_key


def _get_headers() -> dict[str, str]:
    """Get headers for ElevenLabs API requests."""
    return {
        "xi-api-key": _get_api_key(),
        "Content-Type": "application/json",
    }


def load_agent_mappings() -> dict[str, Any]:
    """Load the agent mappings from disk.
    
    Returns:
        Dictionary mapping local agent names to ElevenLabs info.
    """
    if not AGENT_MAPPING_FILE.exists():
        return {}
    
    try:
        with open(AGENT_MAPPING_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load agent mappings: {e}")
        return {}


def save_agent_mappings(mappings: dict[str, Any]) -> None:
    """Save the agent mappings to disk.
    
    Args:
        mappings: Dictionary of agent mappings.
    """
    try:
        with open(AGENT_MAPPING_FILE, "w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"Saved agent mappings to {AGENT_MAPPING_FILE}")
    except IOError as e:
        logger.error(f"Failed to save agent mappings: {e}")


def get_elevenlabs_agent_id(agent_name: str) -> Optional[str]:
    """Get the ElevenLabs agent ID for a local agent.
    
    Args:
        agent_name: Local agent name (e.g., 'meeting_facilitator').
        
    Returns:
        ElevenLabs agent ID if synced, None otherwise.
    """
    mappings = load_agent_mappings()
    agent_info = mappings.get(agent_name)
    if agent_info:
        return agent_info.get("elevenlabs_agent_id")
    return None


async def list_elevenlabs_agents() -> list[dict[str, Any]]:
    """List all agents in ElevenLabs account.
    
    Returns:
        List of agent info dictionaries.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ELEVENLABS_API_BASE}/convai/agents",
                headers=_get_headers(),
                timeout=30.0,
            )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("agents", [])
        else:
            logger.error(f"Failed to list agents: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing ElevenLabs agents: {e}")
        return []


async def get_elevenlabs_agent(agent_id: str) -> Optional[dict[str, Any]]:
    """Get details of a specific ElevenLabs agent.
    
    Args:
        agent_id: The ElevenLabs agent ID.
        
    Returns:
        Agent info if found, None otherwise.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ELEVENLABS_API_BASE}/convai/agents/{agent_id}",
                headers=_get_headers(),
                timeout=30.0,
            )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Agent config: {json.dumps(data, indent=2)}")
            return data
        else:
            logger.error(f"Failed to get agent: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting ElevenLabs agent: {e}")
        return None


def _get_character_customizations(agent_name: str) -> dict[str, Any]:
    """Load character customizations from agent directory.
    
    Character customizations override the base agent config for:
    - voice_id (from voice_id.txt)
    - avatar_name (from avatar_name.txt)
    - custom_prompt (from custom_prompt.txt, appended to system prompt)
    
    Args:
        agent_name: Local agent name.
        
    Returns:
        Dictionary of customizations.
    """
    agent_path = Path(__file__).parent.parent / "agents" / agent_name
    customizations = {}
    
    # Voice ID customization
    voice_file = agent_path / "voice_id.txt"
    if voice_file.exists():
        voice_id = voice_file.read_text().strip()
        if voice_id:
            customizations["voice_id"] = voice_id
            logger.info(f"Using customized voice_id for {agent_name}: {voice_id}")
    
    # Avatar name customization
    name_file = agent_path / "avatar_name.txt"
    if name_file.exists():
        avatar_name = name_file.read_text().strip()
        if avatar_name:
            customizations["avatar_name"] = avatar_name
            logger.info(f"Using customized avatar_name for {agent_name}: {avatar_name}")
    
    # Custom prompt addition
    prompt_file = agent_path / "custom_prompt.txt"
    if prompt_file.exists():
        custom_prompt = prompt_file.read_text().strip()
        if custom_prompt:
            customizations["custom_prompt"] = custom_prompt
            logger.info(f"Using custom_prompt for {agent_name}: {custom_prompt[:50]}...")
    
    return customizations


def _build_agent_payload(agent_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build the API payload for creating/updating an ElevenLabs agent.
    
    Args:
        agent_name: Local agent name.
        config: Agent configuration from YAML.
        
    Returns:
        API payload dictionary.
    """
    elevenlabs_config = config.get("elevenlabs", {})
    
    # Load character customizations (voice, name, custom prompt)
    customizations = _get_character_customizations(agent_name)
    
    # Use prefix to identify our agents
    display_name = f"{AGENT_PREFIX}{agent_name}"
    
    # Build system prompt - base prompt + optional custom prompt
    default_prompt = load_prompt("default_system_prompt") or "You are a helpful assistant."
    system_prompt = elevenlabs_config.get("system_prompt", default_prompt)
    if customizations.get("custom_prompt"):
        system_prompt = f"{system_prompt}\n\n{customizations['custom_prompt']}"
    
    # First message - use avatar name if customized
    first_message = elevenlabs_config.get("first_message", "Hello!")
    if customizations.get("avatar_name"):
        # Optionally personalize first message
        avatar_name = customizations["avatar_name"]
        if "I'm here to help" in first_message:
            first_message = first_message.replace("I'm here to help", f"I'm {avatar_name}, and I'm here to help")
    
    # Voice ID - use customization if available, otherwise config.yaml
    voice_id = customizations.get("voice_id") or elevenlabs_config.get("voice_id", "rachel")
    
    payload = {
        "name": display_name,
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": system_prompt,
                },
                "first_message": first_message,
                "language": "en",
            },
            "tts": {
                "voice_id": voice_id,
            },
        },
    }
    
    # Add tools if defined
    tools = config.get("tools", [])
    if tools:
        # Get the public URL for webhooks
        from utils.ngrok import get_public_url
        public_url = get_public_url()
        
        if not public_url:
            logger.warning("No public URL available (ngrok not running?) - tools will need manual webhook configuration")
        
        tool_configs = []
        for tool in tools:
            if tool == "set_expression":
                tool_config = {
                    "type": "webhook",
                    "name": "set_expression",
                    "description": "Change the avatar's facial expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "enum": config.get("expressions", ["neutral", "happy", "thinking"]),
                                "description": "The expression to show",
                            }
                        },
                        "required": ["expression"],
                    },
                }
                if public_url:
                    tool_config["webhook"] = {"url": f"{public_url}/webhooks/tools/set_expression"}
                tool_configs.append(tool_config)
                
            elif tool == "request_to_speak":
                tool_config = {
                    "type": "webhook",
                    "name": "request_to_speak",
                    "description": "Raise your hand to indicate you want to speak",
                    "parameters": {"type": "object", "properties": {}},
                }
                if public_url:
                    tool_config["webhook"] = {"url": f"{public_url}/webhooks/tools/request_to_speak"}
                tool_configs.append(tool_config)
                
            elif tool == "lower_hand":
                tool_config = {
                    "type": "webhook",
                    "name": "lower_hand", 
                    "description": "Lower your raised hand",
                    "parameters": {"type": "object", "properties": {}},
                }
                if public_url:
                    tool_config["webhook"] = {"url": f"{public_url}/webhooks/tools/lower_hand"}
                tool_configs.append(tool_config)
                
            elif tool == "get_participants":
                tool_config = {
                    "type": "webhook",
                    "name": "get_participants",
                    "description": "Get the list of people currently in the meeting. Use this when you need to know who is present or want to address someone by name.",
                    "parameters": {"type": "object", "properties": {}},
                }
                if public_url:
                    tool_config["webhook"] = {"url": f"{public_url}/webhooks/tools/get_participants"}
                tool_configs.append(tool_config)
                
            elif tool == "get_meeting_context":
                tool_config = {
                    "type": "webhook",
                    "name": "get_meeting_context",
                    "description": "See who's on the call and what they look like - their background, what they're wearing, their setup. Call this immediately when you start to see who you're meeting with!",
                    "parameters": {"type": "object", "properties": {}},
                }
                if public_url:
                    tool_config["webhook"] = {"url": f"{public_url}/webhooks/tools/get_meeting_context"}
                tool_configs.append(tool_config)
        
        if tool_configs:
            payload["conversation_config"]["agent"]["tools"] = tool_configs
            tool_names = [t["name"] for t in tool_configs]
            if public_url:
                logger.info(f"Configured tools: {tool_names} with webhooks at {public_url}")
            else:
                logger.warning(f"Tools configured WITHOUT webhooks (no ngrok URL): {tool_names}")
    
    return payload


async def create_elevenlabs_agent(agent_name: str) -> Optional[str]:
    """Create a new ElevenLabs agent from local config.
    
    Args:
        agent_name: Local agent name (e.g., 'meeting_facilitator').
        
    Returns:
        ElevenLabs agent ID if created, None on failure.
    """
    # Load local config
    config = get_agent_config(agent_name)
    if not config:
        logger.error(f"Agent config not found: {agent_name}")
        return None
    
    payload = _build_agent_payload(agent_name, config)
    
    try:
        logger.info(f"Creating ElevenLabs agent: {AGENT_PREFIX}{agent_name}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ELEVENLABS_API_BASE}/convai/agents/create",
                headers=_get_headers(),
                json=payload,
                timeout=30.0,
            )
        
        if response.status_code in (200, 201):
            data = response.json()
            agent_id = data.get("agent_id")
            
            # Save mapping
            mappings = load_agent_mappings()
            mappings[agent_name] = {
                "elevenlabs_agent_id": agent_id,
                "elevenlabs_name": f"{AGENT_PREFIX}{agent_name}",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            save_agent_mappings(mappings)
            
            logger.info(f"Created ElevenLabs agent: {agent_id}")
            return agent_id
        else:
            logger.error(f"Failed to create agent: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating ElevenLabs agent: {e}")
        return None


async def update_elevenlabs_agent(agent_name: str) -> bool:
    """Update an existing ElevenLabs agent from local config.
    
    Args:
        agent_name: Local agent name.
        
    Returns:
        True if updated successfully, False otherwise.
    """
    # Get existing agent ID
    agent_id = get_elevenlabs_agent_id(agent_name)
    if not agent_id:
        logger.error(f"No ElevenLabs agent found for: {agent_name}")
        return False
    
    # Load local config
    config = get_agent_config(agent_name)
    if not config:
        logger.error(f"Agent config not found: {agent_name}")
        return False
    
    payload = _build_agent_payload(agent_name, config)
    
    try:
        logger.info(f"Updating ElevenLabs agent: {agent_id}")
        
        # Debug: log the tools being sent
        tools = payload.get("conversation_config", {}).get("agent", {}).get("tools", [])
        logger.info(f"Sending {len(tools)} tools to ElevenLabs: {[t.get('name') for t in tools]}")
        
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{ELEVENLABS_API_BASE}/convai/agents/{agent_id}",
                headers=_get_headers(),
                json=payload,
                timeout=30.0,
            )
        
        # Debug: log response
        logger.info(f"ElevenLabs update response: {response.status_code}")
        if response.status_code not in (200, 204):
            logger.error(f"Response body: {response.text}")
        
        if response.status_code in (200, 204):
            # Update mapping timestamp
            mappings = load_agent_mappings()
            if agent_name in mappings:
                mappings[agent_name]["updated_at"] = datetime.utcnow().isoformat()
                save_agent_mappings(mappings)
            
            logger.info(f"Updated ElevenLabs agent: {agent_id}")
            return True
        else:
            logger.error(f"Failed to update agent: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating ElevenLabs agent: {e}")
        return False


async def sync_agent(agent_name: str) -> dict[str, Any]:
    """Sync a local agent to ElevenLabs (create or update).
    
    Args:
        agent_name: Local agent name.
        
    Returns:
        Result dictionary with status and agent_id.
    """
    existing_id = get_elevenlabs_agent_id(agent_name)
    
    if existing_id:
        # Update existing
        success = await update_elevenlabs_agent(agent_name)
        return {
            "action": "updated",
            "success": success,
            "agent_id": existing_id if success else None,
            "agent_name": agent_name,
        }
    else:
        # Create new
        agent_id = await create_elevenlabs_agent(agent_name)
        return {
            "action": "created",
            "success": agent_id is not None,
            "agent_id": agent_id,
            "agent_name": agent_name,
        }


async def list_voices() -> list[dict[str, Any]]:
    """List all available voices in ElevenLabs account.
    
    Returns:
        List of voice info dictionaries.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ELEVENLABS_API_BASE}/voices",
                headers=_get_headers(),
                timeout=30.0,
            )
        
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])
            # Return simplified voice info
            return [
                {
                    "voice_id": v.get("voice_id"),
                    "name": v.get("name"),
                    "category": v.get("category"),
                    "description": v.get("description", "")[:100],
                }
                for v in voices
            ]
        else:
            logger.error(f"Failed to list voices: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing ElevenLabs voices: {e}")
        return []


def get_sync_status() -> list[dict[str, Any]]:
    """Get sync status for all local agents.
    
    Returns:
        List of agent status dictionaries.
    """
    mappings = load_agent_mappings()
    local_agents = list_agents()
    
    result = []
    for agent_name in local_agents:
        mapping = mappings.get(agent_name, {})
        config = get_agent_config(agent_name)
        
        result.append({
            "name": agent_name,
            "display_name": config.get("name", agent_name) if config else agent_name,
            "synced": agent_name in mappings,
            "elevenlabs_agent_id": mapping.get("elevenlabs_agent_id"),
            "elevenlabs_name": mapping.get("elevenlabs_name"),
            "created_at": mapping.get("created_at"),
            "updated_at": mapping.get("updated_at"),
        })
    
    return result
