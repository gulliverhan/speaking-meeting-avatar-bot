"""API routes for the meeting bot application."""

import os
import uuid
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.models import BotRequest, BotResponse, HealthResponse
from core.agent_manager import get_agent_config, list_agents as list_agent_names
from core.bot_state import BOT_STATES, BotState
from core.recall_client import get_recall_client, RecallClient
from utils.logger import logger
from utils.ngrok import LOCAL_DEV_MODE, determine_websocket_url
from utils.prompts import format_prompt, get_expression_modifier, get_animation_prompt


async def wait_for_call_start(
    client_id: str, 
    recall_client: RecallClient, 
    recall_bot_id: str,
    max_attempts: int = 30,
    poll_interval: float = 1.0,
    visual_context_timeout: float = 10.0,
) -> None:
    """Background task to send start_conversation signal when bot joins the call.
    
    Waits for:
    1. Bot to be in_call
    2. Visual context from image analysis (or timeout)
    Then signals the browser to start the conversation.
    
    Args:
        client_id: Our internal bot client ID.
        recall_client: Recall.ai API client.
        recall_bot_id: Recall.ai bot ID.
        max_attempts: Maximum polling attempts for joining.
        poll_interval: Seconds between polls.
        visual_context_timeout: Max seconds to wait for image analysis after joining.
    """
    import asyncio
    
    logger.info(f"Waiting for bot {client_id} to join call...")
    
    in_call_time = None
    
    for attempt in range(max_attempts):
        await asyncio.sleep(poll_interval)
        
        # Check if bot was removed
        if client_id not in BOT_STATES:
            logger.info(f"Bot {client_id} was removed, stopping")
            return
        
        bot_state = BOT_STATES.get(client_id)
        if not bot_state:
            return
            
        try:
            bot_data = await recall_client.get_bot(recall_bot_id)
            if not bot_data:
                continue
            
            # Get current status
            status_changes = bot_data.get("status_changes", [])
            status = status_changes[-1].get("code", "") if status_changes else ""
            
            # Check for terminal states
            if status in ("call_ended", "done", "fatal", "error", "kicked", "left_meeting"):
                logger.info(f"Call ended (status: {status}), cleaning up bot {client_id}")
                BOT_STATES.pop(client_id, None)
                return
            
            # When bot is in the call, start waiting for visual context
            if status in ("in_call_not_recording", "in_call_recording"):
                if in_call_time is None:
                    in_call_time = asyncio.get_event_loop().time()
                    logger.info(f"Bot {client_id} in call, waiting for visual context...")
                
                # Check if we have visual context (participants with descriptions)
                video_participants = bot_state.extra.get("video_participants", {})
                has_visual_context = any(
                    p.get("description") for p in video_participants.values()
                )
                
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - in_call_time
                timed_out = elapsed >= visual_context_timeout
                
                if has_visual_context:
                    logger.info(f"Visual context ready after {elapsed:.1f}s, starting conversation")
                elif timed_out:
                    logger.info(f"Visual context timeout ({visual_context_timeout}s), starting conversation anyway")
                
                # Start conversation if we have context OR timed out
                if (has_visual_context or timed_out) and not bot_state.extra.get("conversation_started"):
                    from app.websockets import BROWSER_CONNECTIONS, send_user_message_to_browser
                    ws = BROWSER_CONNECTIONS.get(client_id)
                    if ws:
                        try:
                            import json
                            
                            # Start the conversation - agent will say "Hey! What's up?"
                            await ws.send_text(json.dumps({"type": "start_conversation"}))
                            logger.info(f"Sent start_conversation to browser {client_id}")
                            bot_state.extra["conversation_started"] = True
                            
                            # If we have visual context, inject it after the initial greeting
                            if video_participants:
                                # Wait for agent's first message to finish (5s for "Hey! What's up?")
                                await asyncio.sleep(5)
                                
                                # Build context about who's in the meeting
                                people_context = []
                                screen_shares = []
                                
                                for stream_key, info in video_participants.items():
                                    name = info.get("name", "Someone")
                                    description = info.get("description", "")
                                    
                                    # Clean up description - remove redundant name at start
                                    # Vision often says "Gulliver Handley is seen..." - we just want the description
                                    clean_desc = description
                                    if clean_desc and clean_desc.lower().startswith(name.lower()):
                                        # Remove "Name is" or "Name," from start (use proper string check)
                                        remainder = clean_desc[len(name):]
                                        # Strip leading space/comma, then "is " if present
                                        remainder = remainder.lstrip(" ,")
                                        if remainder.lower().startswith("is "):
                                            remainder = remainder[3:]
                                        clean_desc = remainder.strip()
                                    
                                    if info.get("is_screen_share"):
                                        # Track screen shares separately
                                        if clean_desc:
                                            screen_shares.append(f"{name} is sharing: {clean_desc}")
                                        else:
                                            screen_shares.append(f"{name} is sharing their screen")
                                    else:
                                        # Regular webcam
                                        if clean_desc:
                                            people_context.append(f"{name} - {clean_desc}")
                                        else:
                                            people_context.append(name)
                                
                                # Send greeting context about people
                                if people_context:
                                    context_msg = (
                                        "[System: People in the meeting: "
                                        + "; ".join(people_context)
                                        + ". Greet them by name and comment on ONE interesting visual detail!]"
                                    )
                                    await send_user_message_to_browser(client_id, context_msg)
                                    logger.info(f"Injected initial context: {context_msg[:100]}...")
                                
                                # If there are screen shares, notify about them after a brief pause
                                if screen_shares:
                                    await asyncio.sleep(2)  # Let greeting happen first
                                    screen_msg = "[System: Screen share active - " + "; ".join(screen_shares) + "]"
                                    await send_user_message_to_browser(client_id, screen_msg)
                                    logger.info(f"Injected screen share context: {screen_msg[:100]}...")
                            
                        except Exception as e:
                            logger.error(f"Failed to send start signal: {e}")
                    return
                
        except Exception as e:
            logger.error(f"Error checking bot status for {client_id}: {e}")
    
    logger.warning(f"Timed out waiting for bot {client_id} to join call")


async def monitor_participant_changes(
    client_id: str,
    recall_client: RecallClient,
    recall_bot_id: str,
    poll_interval: float = 5.0,
) -> None:
    """Monitor for participant join/leave events and inject context.
    
    Also monitors for call ending to clean up bot state.
    
    Args:
        client_id: Our internal bot client ID.
        recall_client: Recall.ai API client.
        recall_bot_id: Recall.ai bot ID.
        poll_interval: Seconds between polls.
    """
    import asyncio
    from app.websockets import send_context_to_browser
    
    logger.info(f"Starting participant monitor for {client_id}")
    
    while client_id in BOT_STATES:
        await asyncio.sleep(poll_interval)
        
        try:
            bot_state = BOT_STATES.get(client_id)
            if not bot_state:
                break
            
            bot_data = await recall_client.get_bot(recall_bot_id)
            if not bot_data:
                continue
            
            # Safely get status
            status_changes = bot_data.get("status_changes", [])
            status = status_changes[-1].get("code", "") if status_changes else ""
            
            # Check for terminal states - clean up bot
            if status in ("call_ended", "done", "fatal"):
                logger.info(f"Call ended (status: {status}), cleaning up bot {client_id}")
                BOT_STATES.pop(client_id, None)
                break
            
            # Get current participants
            participants = bot_data.get("meeting_participants", [])
            current_names = set(p.get("name") for p in participants if p.get("name"))
            known_names = bot_state.extra.get("known_participants", set())
            
            # Detect joins
            joined = current_names - known_names
            if joined:
                for name in joined:
                    context = f"[{name} has joined the meeting]"
                    await send_context_to_browser(client_id, context, speaker="System")
                    logger.info(f"Participant joined: {name}")
            
            # Detect leaves
            left = known_names - current_names
            if left:
                for name in left:
                    context = f"[{name} has left the meeting]"
                    await send_context_to_browser(client_id, context, speaker="System")
                    logger.info(f"Participant left: {name}")
            
            # Update known participants
            bot_state.extra["known_participants"] = current_names
            
        except Exception as e:
            logger.error(f"Error monitoring participants for {client_id}: {e}")
    
    logger.info(f"Participant monitor stopped for {client_id}")

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    """Redirect to admin dashboard."""
    return RedirectResponse(url="/static/admin.html")


@router.post(
    "/bots",
    tags=["bots"],
    response_model=BotResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Bot successfully created and joined the meeting"},
        400: {"description": "Bad request - Missing required fields"},
        500: {"description": "Server error - Failed to create bot"},
    },
)
async def create_bot(request: BotRequest, client_request: Request):
    """Create and deploy a meeting bot.

    Args:
        request: The bot creation request.
        client_request: The FastAPI request object.

    Returns:
        BotResponse with the created bot ID.
    """
    # Validate required parameters
    if not request.meeting_url:
        return JSONResponse(
            content={"message": "Meeting URL is required", "status": "error"},
            status_code=400,
        )

    # Get API key from request state (or use env var)
    api_key = getattr(client_request.state, "api_key", None) or os.getenv(
        "RECALL_API_KEY"
    )

    # Load agent configuration
    agent_config = get_agent_config(request.agent_name) if request.agent_name else None
    
    # Load character customizations (avatar_name, voice_id)
    from pathlib import Path
    agent_path = Path("agents") / request.agent_name if request.agent_name else None
    avatar_name = ""
    if agent_path and (agent_path / "avatar_name.txt").exists():
        avatar_name = (agent_path / "avatar_name.txt").read_text().strip()
    
    # Speaker context: use Recall.ai diarization to identify who is speaking
    use_speaker_context = request.speaker_context
    
    # Audio mode determines where audio processing happens
    use_browser_sdk = request.audio_mode == "browser_sdk"
    use_audio_test = request.audio_mode == "audio_test"

    # Generate a unique client ID for this bot
    bot_client_id = str(uuid.uuid4())

    # Determine WebSocket URL for receiving meeting audio
    websocket_url, _ = determine_websocket_url(None, client_request)

    # Get ElevenLabs agent ID - try synced agent first, then env var
    from core.elevenlabs_agent_sync import get_elevenlabs_agent_id
    elevenlabs_agent_id = get_elevenlabs_agent_id(request.agent_name)
    
    if not elevenlabs_agent_id:
        # Fall back to environment variable
        elevenlabs_agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    
    if not elevenlabs_agent_id:
        return JSONResponse(
            content={
                "message": f"No ElevenLabs agent found. Either sync '{request.agent_name}' to ElevenLabs or set ELEVENLABS_AGENT_ID env var.",
                "status": "error"
            },
            status_code=400,
        )

    # Use bot name: avatar_name > request.bot_name > agent config name > default
    # Priority: Character name (avatar_name.txt) > Explicit request > Agent type name > Default
    bot_name_param = avatar_name or request.bot_name or (agent_config.get("name") if agent_config else None) or "AI Assistant"
    
    # Build HTTP base URL for webhooks and webpage
    http_base = websocket_url.replace("ws://", "http://").replace("wss://", "https://")
    
    # Build the webpage URL based on audio mode
    from urllib.parse import quote
    
    if use_audio_test:
        # Audio test mode: simple page that plays test audio
        # Used to verify Recall.ai audio capture is working
        webpage_url = f"{http_base}/static/audio-test.html"
        audio_ws_url = None
    elif use_browser_sdk:
        # Browser SDK mode: webpage handles audio via ElevenLabs SDK
        # Pass agent_id so the SDK can connect to ElevenLabs
        # Pass agent_name so the webpage loads the correct avatar expressions
        webpage_url = (
            f"{http_base}/static/index.html"
            f"?client_id={bot_client_id}"
            f"&agent_id={elevenlabs_agent_id}"
            f"&agent_name={quote(request.agent_name)}"
            f"&bot_name={quote(bot_name_param)}"
        )
        audio_ws_url = None  # No server-side audio processing
    else:
        # Server mode: server handles audio via WebSocket
        # Webpage only displays avatar (video_only=true)
        # Pass agent_name so the webpage loads the correct avatar expressions
        webpage_url = (
            f"{http_base}/static/index.html"
            f"?client_id={bot_client_id}"
            f"&agent_name={quote(request.agent_name)}"
            f"&bot_name={quote(bot_name_param)}"
            f"&video_only=true"
        )
        # Convert HTTP URL to WebSocket URL (https -> wss, http -> ws)
        ws_base = websocket_url.replace("https://", "wss://").replace("http://", "ws://")
        audio_ws_url = f"{ws_base}/ws/audio/{bot_client_id}"
    
    # Build webhook URL for speaker context (if enabled)
    # Uses Recall.ai's diarization to identify speakers
    transcription_webhook_url = None
    if use_speaker_context:
        transcription_webhook_url = f"{http_base}/webhooks/transcription/{bot_client_id}"

    logger.info(f"Creating bot for meeting: {request.meeting_url}")
    logger.info(f"Agent: {request.agent_name}")
    logger.info(f"Audio mode: {'Browser SDK' if use_browser_sdk else 'Server-side'}")
    logger.info(f"Webpage URL: {webpage_url}")
    if audio_ws_url:
        logger.info(f"Audio WebSocket URL: {audio_ws_url}")
    logger.info(f"Speaker context: {'enabled' if use_speaker_context else 'disabled'}")
    if transcription_webhook_url:
        logger.info(f"Transcription webhook: {transcription_webhook_url}")

    # Create bot state first
    bot_state = BotState(
        client_id=bot_client_id,
        agent_name=request.agent_name,
        meeting_url=request.meeting_url,
        bot_name=bot_name_param,
        elevenlabs_agent_id=elevenlabs_agent_id,
        audio_mode=request.audio_mode,
        speaker_context=use_speaker_context,
    )

    # Call Recall.ai API to create the bot
    # Two modes:
    # - Browser SDK mode: Webpage handles audio via ElevenLabs SDK, Recall captures browser output
    # - Server mode: Server handles audio via WebSocket, separate input/output channels
    try:
        recall_client = get_recall_client(api_key)
        
        # Build output_media configuration
        # Camera: renders our webpage as the bot's video
        # Speaker: captures audio from the webpage (ElevenLabs responses) and sends to meeting
        output_media_config = {
            "camera": {
                "kind": "webpage",
                "config": {
                    "url": webpage_url,
                    "width": 1280,
                    "height": 720,
                }
            }
        }
        
        # In browser SDK or audio test mode, capture webpage audio output for the meeting
        if use_browser_sdk or use_audio_test:
            output_media_config["speaker"] = {
                "kind": "webpage"  # Captures audio from the same webpage
            }
        
        # Configure input media for browser SDK mode
        # This routes meeting audio TO the webpage's microphone (getUserMedia)
        # Critical: Without this, the ElevenLabs SDK can't hear meeting participants!
        input_media_config = None
        if use_browser_sdk:
            input_media_config = {
                "microphone": {
                    "kind": "webpage"  # Routes meeting audio to webpage's getUserMedia
                }
            }
            logger.info("Configured input_media to route meeting audio to webpage microphone")
        
        # Video WebSocket URL for receiving per-participant video frames
        # This gives us participant info (names) even on Google Meet!
        ws_base = http_base.replace("https://", "wss://").replace("http://", "ws://")
        video_ws_url = f"{ws_base}/ws/video/{bot_client_id}"
        logger.info(f"Video WebSocket URL: {video_ws_url}")
        
        recall_response = await recall_client.create_bot(
            meeting_url=request.meeting_url,
            bot_name=bot_name_param,
            bot_image=request.bot_image,
            enable_transcription=use_speaker_context,
            transcription_webhook_url=transcription_webhook_url,
            # Only pass audio WebSocket URL for server mode
            audio_websocket_url=audio_ws_url,
            # Video WebSocket for participant info and optional image analysis
            video_websocket_url=video_ws_url,
            # Webpage for bot display and audio output
            output_media=output_media_config,
            # Route meeting audio to webpage's microphone (for browser SDK mode)
            input_media=input_media_config,
            # Use GPU variant for best output media performance
            # Has 6000 millicores CPU + 13250MB RAM + WebGL support
            variant="web_gpu",
        )

        if recall_response:
            recall_bot_id = recall_response.get("id")
            bot_state.recall_bot_id = recall_bot_id
            BOT_STATES[bot_client_id] = bot_state
            logger.info(f"Recall.ai bot created: {recall_bot_id}")

            # Start background task to signal browser when bot joins call
            import asyncio
            asyncio.create_task(
                wait_for_call_start(bot_client_id, recall_client, recall_bot_id)
            )

            return BotResponse(
                bot_id=bot_client_id,
                agent_name=request.agent_name,
            )
        else:
            return JSONResponse(
                content={
                    "message": "Failed to create bot via Recall.ai",
                    "status": "error",
                },
                status_code=500,
            )

    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        return JSONResponse(
            content={"message": f"Error creating bot: {str(e)}", "status": "error"},
            status_code=500,
        )


@router.delete(
    "/bots/{bot_id}",
    tags=["bots"],
    response_model=dict[str, Any],
    responses={
        200: {"description": "Bot successfully removed"},
        404: {"description": "Bot not found"},
        500: {"description": "Server error - Failed to remove bot"},
    },
)
async def remove_bot(
    bot_id: str,
    client_request: Request,
):
    """Remove a bot from a meeting.

    Args:
        bot_id: The bot ID from the URL path.
        client_request: The FastAPI request object.

    Returns:
        Dictionary with removal status.
    """
    logger.info(f"Removing bot: {bot_id}")

    # Find bot state
    bot_state = BOT_STATES.get(bot_id)

    if not bot_state:
        return JSONResponse(
            content={"message": "Bot not found", "status": "error"},
            status_code=404,
        )

    success = True

    # Call Recall.ai API to remove the bot
    if bot_state.recall_bot_id:
        try:
            api_key = getattr(client_request.state, "api_key", None) or os.getenv(
                "RECALL_API_KEY"
            )
            recall_client = get_recall_client(api_key)
            if not await recall_client.delete_bot(bot_state.recall_bot_id):
                success = False
                logger.warning("Failed to delete bot from Recall.ai")
        except Exception as e:
            logger.error(f"Error deleting Recall.ai bot: {e}")
            success = False

    # Clean up ElevenLabs bridge
    from core.elevenlabs_client import remove_bridge

    try:
        await remove_bridge(bot_id)
    except Exception as e:
        logger.error(f"Error removing ElevenLabs bridge: {e}")

    # Remove bot state
    BOT_STATES.pop(bot_id, None)

    logger.info(f"Bot {bot_id} removed {'successfully' if success else 'with errors'}")

    return {
        "message": "Bot removed successfully" if success else "Bot removed with errors",
        "status": "success" if success else "partial",
        "bot_id": bot_id,
    }


@router.get(
    "/bots",
    tags=["bots"],
    response_model=list[dict[str, Any]],
)
async def list_bots(client_request: Request):
    """List all active bots.
    
    Also checks bot status from Recall.ai and auto-removes ended bots.

    Returns:
        List of active bot states.
    """
    api_key = getattr(client_request.state, "api_key", None) or os.getenv("RECALL_API_KEY")
    recall_client = get_recall_client(api_key)
    
    # Check status of each bot and clean up ended ones
    bots_to_remove = []
    
    # Create a snapshot to avoid "dictionary changed size during iteration" errors
    for bot_id, state in list(BOT_STATES.items()):
        if state.recall_bot_id:
            try:
                bot_data = await recall_client.get_bot(state.recall_bot_id)
                if bot_data:
                    status_changes = bot_data.get("status_changes", [])
                    current_status = status_changes[-1].get("code", "") if status_changes else ""
                    
                    # Terminal states that mean the bot is no longer active
                    if current_status in ("call_ended", "done", "fatal", "error", "kicked", "left_meeting"):
                        logger.info(f"Bot {bot_id} ended (status: {current_status}), marking for cleanup")
                        bots_to_remove.append(bot_id)
            except Exception as e:
                logger.debug(f"Error checking bot {bot_id} status: {e}")
    
    # Remove ended bots
    for bot_id in bots_to_remove:
        BOT_STATES.pop(bot_id, None)
        logger.info(f"Auto-cleaned up ended bot: {bot_id}")
    
    return [
        {
            "bot_id": bot_id,
            "bot_name": state.bot_name,
            "agent_name": state.agent_name,
            "meeting_url": state.meeting_url,
            "current_expression": state.current_expression,
            "wants_to_speak": state.wants_to_speak,
            "audio_mode": state.audio_mode,
            "speaker_context": state.speaker_context,
        }
        for bot_id, state in BOT_STATES.items()
    ]


@router.get(
    "/health",
    tags=["system"],
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """Health check endpoint.

    Returns:
        Health status.
    """
    return HealthResponse(status="healthy", version="0.1.0")


@router.get(
    "/bots/{bot_id}/participants",
    tags=["bots"],
)
async def get_bot_participants(bot_id: str, client_request: Request):
    """Get participants in the meeting for a specific bot.

    Args:
        bot_id: The bot's client ID.
        client_request: The FastAPI request object.

    Returns:
        List of participants with their names and info.
    """
    # Find bot state
    bot_state = BOT_STATES.get(bot_id)
    if not bot_state:
        return JSONResponse(
            content={"message": "Bot not found", "status": "error"},
            status_code=404,
        )

    if not bot_state.recall_bot_id:
        return JSONResponse(
            content={"message": "Bot not connected to Recall.ai", "status": "error"},
            status_code=400,
        )

    try:
        api_key = getattr(client_request.state, "api_key", None) or os.getenv(
            "RECALL_API_KEY"
        )
        recall_client = get_recall_client(api_key)
        participants = await recall_client.get_meeting_participants(bot_state.recall_bot_id)
        
        # Simplify participant data for response
        simplified = []
        for p in participants:
            simplified.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "is_host": p.get("is_host", False),
                "platform": p.get("platform"),
            })
        
        return {
            "bot_id": bot_id,
            "participants": simplified,
            "count": len(simplified),
        }
    except Exception as e:
        logger.error(f"Error getting participants: {e}")
        return JSONResponse(
            content={"message": f"Error: {str(e)}", "status": "error"},
            status_code=500,
        )


@router.post(
    "/bots/{bot_id}/announce-participants",
    tags=["bots"],
)
async def announce_participants(bot_id: str, client_request: Request):
    """Get participants and send context to the AI agent about who's in the meeting.

    This injects a contextual update like:
    "Current meeting participants: Billy (host), John, Fred"

    Args:
        bot_id: The bot's client ID.
        client_request: The FastAPI request object.

    Returns:
        Success status and list of participants.
    """
    from app.websockets import send_context_to_browser
    
    # Find bot state
    bot_state = BOT_STATES.get(bot_id)
    if not bot_state:
        return JSONResponse(
            content={"message": "Bot not found", "status": "error"},
            status_code=404,
        )

    if not bot_state.recall_bot_id:
        return JSONResponse(
            content={"message": "Bot not connected to Recall.ai", "status": "error"},
            status_code=400,
        )

    try:
        api_key = getattr(client_request.state, "api_key", None) or os.getenv(
            "RECALL_API_KEY"
        )
        recall_client = get_recall_client(api_key)
        participants = await recall_client.get_meeting_participants(bot_state.recall_bot_id)
        
        # Build participant list message
        names = []
        for p in participants:
            name = p.get("name", "Unknown")
            if p.get("is_host"):
                name += " (host)"
            names.append(name)
        
        if names:
            context = f"[Meeting participants: {', '.join(names)}]"
        else:
            context = "[No other participants detected in the meeting yet]"
        
        # Send context to browser for injection into ElevenLabs
        sent = await send_context_to_browser(bot_id, context, speaker="System")
        
        logger.info(f"Announced participants: {context}")
        
        return {
            "success": sent,
            "participants": names,
            "context_sent": context,
        }
    except Exception as e:
        logger.error(f"Error announcing participants: {e}")
        return JSONResponse(
            content={"message": f"Error: {str(e)}", "status": "error"},
            status_code=500,
        )


@router.get(
    "/agents",
    tags=["agents"],
    response_model=list[str],
)
async def list_agents():
    """List available agent configurations.

    Returns:
        List of available agent names.
    """
    return list_agent_names()


# =============================================================================
# Agent Sync Endpoints (ElevenLabs Agent Management)
# =============================================================================

@router.get(
    "/agents/sync-status",
    tags=["agents"],
)
async def get_agents_sync_status():
    """Get sync status for all local agents.

    Returns:
        List of agent sync status objects.
    """
    from core.elevenlabs_agent_sync import get_sync_status
    return get_sync_status()


@router.post(
    "/agents/{agent_name}/sync",
    tags=["agents"],
)
async def sync_agent(agent_name: str):
    """Sync a local agent to ElevenLabs (create or update).

    Args:
        agent_name: Name of the local agent to sync.

    Returns:
        Sync result with action taken and agent ID.
    """
    from core.elevenlabs_agent_sync import sync_agent as do_sync
    result = await do_sync(agent_name)
    
    if not result["success"]:
        return JSONResponse(
            content={"message": f"Failed to sync agent: {agent_name}", "result": result},
            status_code=500,
        )
    
    return result


@router.get(
    "/agents/elevenlabs",
    tags=["agents"],
)
async def list_elevenlabs_agents():
    """List all agents in the ElevenLabs account.

    Returns:
        List of ElevenLabs agents.
    """
    from core.elevenlabs_agent_sync import list_elevenlabs_agents as list_el_agents
    agents = await list_el_agents()
    return {"agents": agents}


@router.get(
    "/voices",
    tags=["agents"],
)
async def list_voices():
    """List all available voices in ElevenLabs.

    Returns:
        List of available voices with IDs and names.
    """
    from core.elevenlabs_agent_sync import list_voices as get_voices
    voices = await get_voices()
    return {"voices": voices}


@router.post(
    "/voices/{voice_id}/preview",
    tags=["agents"],
)
async def preview_voice(voice_id: str, request: Request):
    """Generate a voice preview using ElevenLabs TTS.

    Args:
        voice_id: ElevenLabs voice ID.
        request: Request with text to speak.

    Returns:
        Audio file (MP3).
    """
    from fastapi.responses import Response
    import httpx
    
    try:
        body = await request.json()
        text = body.get("text", "Hello, this is a voice preview.")
    except Exception:
        text = "Hello, this is a voice preview."
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return JSONResponse(
            content={"message": "ELEVENLABS_API_KEY not set"},
            status_code=500,
        )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"ElevenLabs TTS error: {error_text}")
                return JSONResponse(
                    content={"message": f"TTS error: {error_text}"},
                    status_code=response.status_code,
                )
            
            # Return audio as MP3
            return Response(
                content=response.content,
                media_type="audio/mpeg",
            )
            
    except Exception as e:
        logger.error(f"Error generating voice preview: {e}")
        return JSONResponse(
            content={"message": f"Error: {str(e)}"},
            status_code=500,
        )


@router.get(
    "/agents/{agent_name}/elevenlabs-config",
    tags=["agents"],
)
async def get_agent_elevenlabs_config(agent_name: str):
    """Get the full ElevenLabs configuration for a synced agent.

    Args:
        agent_name: Local agent name.

    Returns:
        Full ElevenLabs agent configuration.
    """
    from core.elevenlabs_agent_sync import get_elevenlabs_agent_id, get_elevenlabs_agent
    
    agent_id = get_elevenlabs_agent_id(agent_name)
    if not agent_id:
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not synced to ElevenLabs"},
            status_code=404,
        )
    
    config = await get_elevenlabs_agent(agent_id)
    if not config:
        return JSONResponse(
            content={"message": "Failed to fetch agent config from ElevenLabs"},
            status_code=500,
        )
    
    return config


# =============================================================================
# Avatar Generation Endpoints
# =============================================================================

@router.get(
    "/agents/{agent_name}/expressions-list",
    tags=["agents"],
)
async def get_agent_expressions(agent_name: str):
    """Get the list of expressions for an agent, including animation status.

    Args:
        agent_name: Name of the agent.

    Returns:
        List of expression names and which have idle/speaking animations.
    """
    from pathlib import Path
    
    if agent_name not in list_agent_names():
        return {
            "expressions": ["neutral"],
            "idle_animations": {},
            "speaking_animations": {},
        }  # Fallback
    
    agent_config = get_agent_config(agent_name)
    expressions = agent_config.get("expressions", ["neutral", "happy", "thinking", "interested"])
    
    # Filter out "speaking" from expressions list (it's no longer a separate expression)
    expressions = [e for e in expressions if e != "speaking"]
    
    # Check which expressions have animated GIFs
    expressions_path = Path("agents") / agent_name / "expressions"
    idle_animations = {}
    speaking_animations = {}
    for expr in expressions:
        idle_gif_path = expressions_path / f"{expr}.gif"
        speaking_gif_path = expressions_path / f"{expr}_speaking.gif"
        idle_animations[expr] = idle_gif_path.exists()
        speaking_animations[expr] = speaking_gif_path.exists()
    
    return {
        "expressions": expressions,
        "idle_animations": idle_animations,
        "speaking_animations": speaking_animations,
    }


@router.get(
    "/agents/avatar-status",
    tags=["agents"],
)
async def get_agents_avatar_status():
    """Get avatar/expression status for all local agents.

    Returns:
        List of agents with their expression images status.
    """
    from pathlib import Path
    
    agents_dir = Path("agents")
    result = []
    
    for agent_name in list_agent_names():
        agent_config = get_agent_config(agent_name)
        agent_path = agents_dir / agent_name
        expressions_path = agent_path / "expressions"
        
        # Get list of expressions from config or default
        expressions = agent_config.get("expressions", ["neutral", "happy", "thinking", "interested"])
        
        # Filter out "speaking" from expressions list (it's no longer a separate expression)
        expressions = [e for e in expressions if e != "speaking"]
        
        # Check which expression images and animations exist
        expression_images = {}
        expression_idle_animations = {}
        expression_speaking_animations = {}
        has_base = False
        for expr in expressions:
            img_path = expressions_path / f"{expr}.png"
            idle_gif_path = expressions_path / f"{expr}.gif"
            speaking_gif_path = expressions_path / f"{expr}_speaking.gif"
            if img_path.exists():
                expression_images[expr] = True
                if expr == "neutral":
                    has_base = True
            else:
                expression_images[expr] = False
            # Check for idle and speaking animation GIFs
            expression_idle_animations[expr] = idle_gif_path.exists()
            expression_speaking_animations[expr] = speaking_gif_path.exists()
        
        # Check if base.png exists (the reference image for expressions)
        base_path = expressions_path / "base.png"
        has_base = has_base or base_path.exists()
        
        # Build base image URL if it exists (with cache-buster)
        base_image_url = None
        if base_path.exists():
            import os
            mtime = int(os.path.getmtime(base_path))
            base_image_url = f"/agents/{agent_name}/expressions/base.png?t={mtime}"
        
        # Load saved prompt if exists
        prompt_file = agent_path / "avatar_prompt.txt"
        avatar_prompt = prompt_file.read_text().strip() if prompt_file.exists() else ""
        
        # Load saved avatar name if exists
        name_file = agent_path / "avatar_name.txt"
        avatar_name = name_file.read_text().strip() if name_file.exists() else ""
        
        # Load saved voice ID if exists
        voice_file = agent_path / "voice_id.txt"
        voice_id = voice_file.read_text().strip() if voice_file.exists() else ""
        
        result.append({
            "name": agent_name,
            "display_name": agent_config.get("name", agent_name.replace("_", " ").title()),
            "expressions": expressions,
            "expression_images": expression_images,
            "expression_idle_animations": expression_idle_animations,  # Track which have idle GIFs
            "expression_speaking_animations": expression_speaking_animations,  # Track which have speaking GIFs
            "has_base_avatar": has_base,
            "base_image_url": base_image_url,  # URL to display base image
            "avatar_prompt": avatar_prompt,
            "avatar_name": avatar_name,
            "voice_id": voice_id,
        })
    
    return result


@router.post(
    "/agents/{agent_name}/upload-reference",
    tags=["agents"],
)
async def upload_reference_image(agent_name: str, request: Request):
    """Upload a reference image for avatar generation.

    Args:
        agent_name: Name of the agent.
        request: Multipart form request with file.

    Returns:
        URL of the uploaded image.
    """
    from pathlib import Path
    import base64
    from fastapi import UploadFile
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    # Parse multipart form
    form = await request.form()
    file = form.get("file")
    
    if not file:
        return JSONResponse(
            content={"message": "No file uploaded"},
            status_code=400,
        )
    
    # Read file content
    content = await file.read()
    
    # Save to agent's directory
    agent_path = Path("agents") / agent_name
    ref_path = agent_path / "reference.png"
    ref_path.write_bytes(content)
    
    # Return URL that can be used to access it
    # For Replicate, we need a publicly accessible URL
    # We'll serve it via our own endpoint
    url = f"/agents/{agent_name}/reference.png"
    
    logger.info(f"Reference image uploaded for {agent_name}: {len(content)} bytes")
    
    return {
        "success": True,
        "url": url,
        "size": len(content),
    }


@router.get(
    "/agents/{agent_name}/reference.png",
    tags=["agents"],
)
async def get_reference_image(agent_name: str):
    """Serve a reference image for an agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        The reference image file.
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    image_path = Path("agents") / agent_name / "reference.png"
    
    if not image_path.exists():
        return JSONResponse(
            content={"message": "Reference image not found"},
            status_code=404,
        )
    
    return FileResponse(image_path, media_type="image/png")


@router.post(
    "/agents/{agent_name}/save-settings",
    tags=["agents"],
)
async def save_agent_settings(agent_name: str, request: Request):
    """Save agent settings (name, voice) without regenerating images.

    Args:
        agent_name: Name of the agent.
        request: Request with avatar_name and voice_id.

    Returns:
        Success status.
    """
    from pathlib import Path
    
    try:
        body = await request.json()
        avatar_name = body.get("avatar_name", "")
        voice_id = body.get("voice_id", "")
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    try:
        agent_path = Path("agents") / agent_name
        
        # Save avatar name
        if avatar_name:
            (agent_path / "avatar_name.txt").write_text(avatar_name)
        elif (agent_path / "avatar_name.txt").exists():
            (agent_path / "avatar_name.txt").unlink()
        
        # Save voice ID
        if voice_id:
            (agent_path / "voice_id.txt").write_text(voice_id)
        elif (agent_path / "voice_id.txt").exists():
            (agent_path / "voice_id.txt").unlink()
        
        logger.info(f"Saved settings for {agent_name}: name={avatar_name}, voice={voice_id}")
        
        return {
            "success": True,
            "avatar_name": avatar_name,
            "voice_id": voice_id,
        }
        
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return JSONResponse(
            content={"message": f"Error saving settings: {str(e)}"},
            status_code=500,
        )


@router.post(
    "/agents/{agent_name}/save-base-avatar",
    tags=["agents"],
)
async def save_base_avatar(agent_name: str, request: Request):
    """Save the approved base avatar for use as reference in expression generation.

    Args:
        agent_name: Name of the agent.
        request: Request with image_url, avatar_name, and prompt.

    Returns:
        Local path to saved image.
    """
    from pathlib import Path
    import httpx
    
    try:
        body = await request.json()
        image_url = body.get("image_url", "")
        avatar_name = body.get("avatar_name", "")
        prompt = body.get("prompt", "")
        voice_id = body.get("voice_id", "")
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    if not image_url:
        return JSONResponse(
            content={"message": "image_url is required"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    try:
        # Download the image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_data = response.content
        
        # Save to agent's directory
        agent_path = Path("agents") / agent_name
        expressions_path = agent_path / "expressions"
        expressions_path.mkdir(parents=True, exist_ok=True)
        
        # Save as base.png (the reference for expressions)
        base_path = expressions_path / "base.png"
        base_path.write_bytes(image_data)
        
        # Also save as neutral.png if it doesn't exist
        neutral_path = expressions_path / "neutral.png"
        if not neutral_path.exists():
            neutral_path.write_bytes(image_data)
        
        # Save avatar name, prompt, and voice for reference
        if avatar_name:
            (agent_path / "avatar_name.txt").write_text(avatar_name)
        if prompt:
            (agent_path / "avatar_prompt.txt").write_text(prompt)
        if voice_id:
            (agent_path / "voice_id.txt").write_text(voice_id)
        
        logger.info(f"Saved base avatar for {agent_name}: {base_path}, voice: {voice_id}")
        
        return {
            "success": True,
            "local_path": str(base_path),
            "avatar_name": avatar_name,
            "voice_id": voice_id,
        }
        
    except Exception as e:
        logger.error(f"Error saving base avatar: {e}")
        return JSONResponse(
            content={"message": f"Error saving base avatar: {str(e)}"},
            status_code=500,
        )


@router.post(
    "/agents/{agent_name}/generate-avatar",
    tags=["agents"],
)
async def generate_avatar(agent_name: str, request: Request):
    """Generate a base avatar image for an agent.

    Args:
        agent_name: Name of the agent.
        request: Request with prompt, model_id, and optional reference_image_url in body.

    Returns:
        Generated image URL.
    """
    from utils.image_generator import generate_avatar_image
    from pathlib import Path
    
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model_id = body.get("model_id", "p-image")  # Default to fast/cheap model
        reference_image_url = body.get("reference_image_url")  # Optional
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    if not prompt:
        return JSONResponse(
            content={"message": "Prompt is required"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    # If reference image URL is a local path, convert to full URL for Replicate
    # Replicate needs a publicly accessible URL
    full_ref_url = None
    if reference_image_url:
        if reference_image_url.startswith("/"):
            # It's a local path - construct full public URL using ngrok
            from utils.ngrok import get_public_url
            public_base = get_public_url()
            if public_base:
                full_ref_url = f"{public_base}{reference_image_url}"
                logger.info(f"Converted local path to public URL: {full_ref_url}")
            else:
                logger.warning("No public URL available (ngrok not running?). Reference image may not work.")
                full_ref_url = None
        else:
            full_ref_url = reference_image_url
    
    logger.info(f"Generating avatar for {agent_name} using {model_id}: {prompt[:50]}...")
    if full_ref_url:
        logger.info(f"Using reference image: {full_ref_url[:50]}...")
    
    try:
        result = await generate_avatar_image(prompt, model_id=model_id, reference_image_url=full_ref_url)
        
        if result.get("success"):
            # Save the prompt for future reference
            agent_path = Path("agents") / agent_name
            prompt_file = agent_path / "avatar_prompt.txt"
            prompt_file.write_text(prompt)
            
            return {
                "success": True,
                "image_url": result["image_url"],
                "prompt": prompt,
                "model": model_id,
            }
        else:
            return JSONResponse(
                content={"message": result.get("error", "Generation failed")},
                status_code=500,
            )
    except Exception as e:
        logger.error(f"Error generating avatar: {e}")
        return JSONResponse(
            content={"message": f"Error generating avatar: {str(e)}"},
            status_code=500,
        )


@router.post(
    "/agents/{agent_name}/generate-expressions",
    tags=["agents"],
)
async def generate_expressions(agent_name: str, request: Request):
    """Generate all expression images for an agent based on a base avatar.

    Args:
        agent_name: Name of the agent.
        request: Request with base_image_url, prompt, model_id, avatar_name, use_base_as_reference.

    Returns:
        List of generated expression images.
    """
    from utils.image_generator import generate_expression_pack
    from utils.ngrok import get_public_url
    from pathlib import Path
    
    try:
        body = await request.json()
        base_image_url = body.get("base_image_url", "")
        prompt = body.get("prompt", "")
        model_id = body.get("model_id", "p-image")
        avatar_name = body.get("avatar_name", "")
        use_base_as_reference = body.get("use_base_as_reference", False)
        background_type = body.get("background_type", "none")  # none, match_base, custom
        custom_background = body.get("custom_background", "")
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    if not prompt:
        return JSONResponse(
            content={"message": "prompt is required"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    # Get expressions from agent config
    agent_config = get_agent_config(agent_name)
    expressions = agent_config.get("expressions", ["neutral", "happy", "thinking", "interested"])
    
    # Build reference image URL from saved base avatar
    reference_image_url = None
    if use_base_as_reference:
        base_path = Path("agents") / agent_name / "expressions" / "base.png"
        if base_path.exists():
            public_url = get_public_url()
            if public_url:
                reference_image_url = f"{public_url}/agents/{agent_name}/expressions/base.png"
                logger.info(f"Using saved base avatar as reference: {reference_image_url}")
            else:
                logger.warning("No public URL available for base avatar reference")
    
    # Build background instruction based on settings
    if background_type == "custom" and custom_background:
        background_instruction = custom_background
    elif background_type == "match_base" and reference_image_url:
        background_instruction = "match_base"  # Special flag for generate_expression_pack
    else:
        background_instruction = None  # Default: clean neutral background
    
    logger.info(f"Background setting: {background_type}")
    
    # Enhance prompt with avatar name if provided
    enhanced_prompt = prompt
    if avatar_name:
        enhanced_prompt = f"{avatar_name}: {prompt}"
    
    logger.info(f"Generating expression pack for {agent_name} using {model_id}: {expressions}")
    
    try:
        # Create expressions directory
        expressions_dir = Path("agents") / agent_name / "expressions"
        expressions_dir.mkdir(parents=True, exist_ok=True)
        
        result = await generate_expression_pack(
            base_prompt=enhanced_prompt,
            base_image_url=base_image_url,
            expressions=expressions,
            output_dir=expressions_dir,
            model_id=model_id,
            reference_image_url=reference_image_url,
            background_instruction=background_instruction,
        )
        
        if result.get("success"):
            return {
                "success": True,
                "expressions": result["expressions"],
                "saved_to": str(expressions_dir),
                "model": model_id,
                "generated": result.get("generated", 0),
                "failed": result.get("failed", 0),
            }
        else:
            return JSONResponse(
                content={"message": result.get("error", "Generation failed"), "errors": result.get("errors")},
                status_code=500,
            )
    except Exception as e:
        logger.error(f"Error generating expressions: {e}")
        return JSONResponse(
            content={"message": f"Error generating expressions: {str(e)}"},
            status_code=500,
        )


@router.post(
    "/agents/{agent_name}/generate-expression",
    tags=["agents"],
)
async def generate_single_expression(agent_name: str, request: Request):
    """Generate a single expression image for an agent.

    Args:
        agent_name: Name of the agent.
        request: Request with expression, prompt, custom_modifier, model_id, avatar_name, use_base_as_reference.

    Returns:
        Generated expression info.
    """
    from utils.image_generator import generate_avatar_image
    from utils.ngrok import get_public_url
    from pathlib import Path
    import httpx
    
    try:
        body = await request.json()
        expression = body.get("expression", "")
        base_prompt = body.get("prompt", "")
        custom_modifier = body.get("custom_modifier")  # Optional override
        model_id = body.get("model_id", "p-image")
        avatar_name = body.get("avatar_name", "")
        use_base_as_reference = body.get("use_base_as_reference", False)
        background_type = body.get("background_type", "none")  # none, match_base, custom
        custom_background = body.get("custom_background", "")
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    if not expression or not base_prompt:
        return JSONResponse(
            content={"message": "expression and prompt are required"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    # Build reference image URL from saved base avatar
    reference_image_url = None
    if use_base_as_reference:
        base_path = Path("agents") / agent_name / "expressions" / "base.png"
        if base_path.exists():
            public_url = get_public_url()
            if public_url:
                reference_image_url = f"{public_url}/agents/{agent_name}/expressions/base.png"
                logger.info(f"Using saved base avatar as reference: {reference_image_url}")
    
    # Build background instruction based on settings
    if background_type == "custom" and custom_background:
        background_instruction = f"Background: {custom_background}"
    elif background_type == "match_base" and reference_image_url:
        background_instruction = "Background: Keep the EXACT same background as the reference image."
    else:
        # Default: let the model decide (clean neutral background)
        background_instruction = "Background: Clean neutral background."
    
    logger.info(f"Background setting: {background_type} -> {background_instruction[:50]}...")
    
    # Get expression modifier from prompt files
    if custom_modifier:
        modifier = custom_modifier
    else:
        modifier = get_expression_modifier(expression)
    
    # Enhance prompt with avatar name and reference image context
    if reference_image_url:
        # When using reference image, focus on keeping the same person
        person_desc = f"{avatar_name} from the reference image" if avatar_name else "the person from the reference image"
        full_prompt = format_prompt(
            "single_expression_from_reference",
            person_desc=person_desc,
            modifier=modifier,
            expression=expression,
            background_instruction=background_instruction,
        )
    else:
        person_desc = f"{avatar_name}: {base_prompt}" if avatar_name else base_prompt
        full_prompt = format_prompt(
            "single_expression_from_scratch",
            person_desc=person_desc,
            modifier=modifier,
            background_instruction=background_instruction,
        )
    
    # Fallback if prompt files are missing
    if not full_prompt:
        full_prompt = f"Professional headshot portrait of {person_desc}, {modifier}"
    
    logger.info(f"Generating {expression} for {agent_name} using {model_id}")
    if reference_image_url:
        logger.info(f"With reference image: {reference_image_url[:50]}...")
    
    try:
        result = await generate_avatar_image(full_prompt, model_id=model_id, reference_image_url=reference_image_url)
        
        if result.get("success"):
            # Download and save the image
            image_url = result["image_url"]
            expressions_dir = Path("agents") / agent_name / "expressions"
            expressions_dir.mkdir(parents=True, exist_ok=True)
            image_path = expressions_dir / f"{expression}.png"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image_path.write_bytes(response.content)
            
            logger.info(f"Saved {expression} to {image_path}")
            
            return {
                "success": True,
                "expression": expression,
                "image_path": str(image_path),
                "image_url": image_url,
            }
        else:
            return JSONResponse(
                content={"message": result.get("error", "Generation failed")},
                status_code=500,
            )
    except Exception as e:
        logger.error(f"Error generating {expression}: {e}")
        return JSONResponse(
            content={"message": f"Error generating expression: {str(e)}"},
            status_code=500,
        )


@router.get(
    "/agents/{agent_name}/expressions/{expression}.png",
    tags=["agents"],
)
async def get_expression_image(agent_name: str, expression: str):
    """Serve an expression image for an agent.

    Args:
        agent_name: Name of the agent.
        expression: Expression name (e.g., "neutral", "happy").

    Returns:
        The expression image file.
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    image_path = Path("agents") / agent_name / "expressions" / f"{expression}.png"
    
    if not image_path.exists():
        return JSONResponse(
            content={"message": "Expression image not found"},
            status_code=404,
        )
    
    return FileResponse(image_path, media_type="image/png")


# =============================================================================
# Animation Endpoints
# =============================================================================

@router.get(
    "/agents/{agent_name}/expressions/{expression}.gif",
    tags=["agents"],
)
async def get_expression_animation(agent_name: str, expression: str):
    """Serve an animated GIF for an expression.

    Args:
        agent_name: Name of the agent.
        expression: Expression name (e.g., "neutral", "happy").

    Returns:
        The animated GIF file.
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    gif_path = Path("agents") / agent_name / "expressions" / f"{expression}.gif"
    
    if not gif_path.exists():
        return JSONResponse(
            content={"message": "Animation not found"},
            status_code=404,
        )
    
    return FileResponse(gif_path, media_type="image/gif")


def _load_agent_animation_prompts(agent_name: str) -> dict:
    """Load custom animation prompts for an agent."""
    import yaml
    from pathlib import Path
    
    prompts_path = Path("agents") / agent_name / "custom_animation_prompts.yaml"
    if prompts_path.exists():
        try:
            return yaml.safe_load(prompts_path.read_text()) or {}
        except Exception:
            return {}
    return {}


def _save_agent_animation_prompt(agent_name: str, expression: str, prompt: str) -> None:
    """Save a custom animation prompt for an agent's expression."""
    import yaml
    from pathlib import Path
    
    prompts_path = Path("agents") / agent_name / "custom_animation_prompts.yaml"
    prompts = _load_agent_animation_prompts(agent_name)
    prompts[expression] = prompt
    prompts_path.write_text(yaml.dump(prompts, default_flow_style=False))


def _get_speaking_prompt_from_idle(idle_prompt: str, expression: str) -> str:
    """Convert an idle prompt to a speaking prompt by adding speaking modifiers."""
    # Get the speaking-specific additions from the default prompts
    default_idle = get_animation_prompt(expression, "idle")
    default_speaking = get_animation_prompt(expression, "speaking")
    
    # If they customized the idle prompt, append speaking modifiers
    if idle_prompt != default_idle:
        # Add speaking motion to their custom prompt
        return f"{idle_prompt}, talking with natural mouth movements and speech gestures"
    else:
        # Use the default speaking prompt
        return default_speaking


@router.post(
    "/agents/{agent_name}/generate-animation",
    tags=["agents"],
)
async def generate_expression_animation_endpoint(agent_name: str, request: Request):
    """Generate an animated GIF for an expression.

    Uses bytedance/seedance-1-pro-fast to create an animation from
    a static expression image, then converts to a looping GIF.

    Args:
        agent_name: Name of the agent.
        request: Request with expression name, animation_type (idle/speaking), prompt, and duration.

    Returns:
        Generated animation info.
    """
    from utils.image_generator import generate_expression_animation
    from utils.ngrok import get_public_url
    from pathlib import Path
    
    try:
        body = await request.json()
        expression = body.get("expression", "")
        animation_type = body.get("animation_type", "idle")  # "idle" or "speaking"
        prompt = body.get("prompt", "")
        duration = body.get("duration", 2)  # Default 2 seconds
    except Exception:
        return JSONResponse(
            content={"message": "Invalid request body"},
            status_code=400,
        )
    
    if not expression:
        return JSONResponse(
            content={"message": "expression is required"},
            status_code=400,
        )
    
    if animation_type not in ("idle", "speaking"):
        return JSONResponse(
            content={"message": "animation_type must be 'idle' or 'speaking'"},
            status_code=400,
        )
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    # Check source image exists
    image_path = Path("agents") / agent_name / "expressions" / f"{expression}.png"
    if not image_path.exists():
        return JSONResponse(
            content={"message": f"Expression image '{expression}.png' not found. Generate the static image first."},
            status_code=400,
        )
    
    # Get public URL for Replicate to access our images
    public_url = get_public_url()
    if not public_url:
        return JSONResponse(
            content={"message": "No public URL available (ngrok not running?). Animation generation requires a public URL."},
            status_code=500,
        )
    
    # Use default prompt if not provided
    if not prompt:
        prompt = get_animation_prompt(expression, animation_type)
    
    # Save custom prompt when generating idle animation (for later speaking derivation)
    if animation_type == "idle" and prompt:
        _save_agent_animation_prompt(agent_name, expression, prompt)
        logger.info(f"Saved custom idle prompt for {agent_name}/{expression}")
    
    # Output path depends on animation type
    if animation_type == "speaking":
        output_gif_path = Path("agents") / agent_name / "expressions" / f"{expression}_speaking.gif"
    else:
        output_gif_path = Path("agents") / agent_name / "expressions" / f"{expression}.gif"
    
    logger.info(f"Generating {duration}s {animation_type} animation for {agent_name}/{expression}: {prompt[:50]}...")
    
    try:
        result = await generate_expression_animation(
            image_path=image_path,
            output_gif_path=output_gif_path,
            prompt=prompt,
            public_base_url=public_url,
            duration=duration,
        )
        
        if result.get("success"):
            return {
                "success": True,
                "expression": expression,
                "animation_type": animation_type,
                "gif_path": str(output_gif_path),
                "prompt": prompt,
            }
        else:
            return JSONResponse(
                content={"message": result.get("error", "Animation generation failed")},
                status_code=500,
            )
    except Exception as e:
        logger.error(f"Error generating animation: {e}")
        return JSONResponse(
            content={"message": f"Error generating animation: {str(e)}"},
            status_code=500,
        )


@router.get(
    "/agents/{agent_name}/expressions/{expression}/animation-prompt",
    tags=["agents"],
)
async def get_expression_animation_prompt_endpoint(agent_name: str, expression: str, animation_type: str = "idle"):
    """Get the animation prompt for an expression.

    For idle animations, returns saved custom prompt or default.
    For speaking animations, derives from saved idle prompt + speaking modifiers.

    Args:
        agent_name: Name of the agent.
        expression: Expression name.
        animation_type: Either 'idle' or 'speaking'.

    Returns:
        Animation prompt for this expression and type.
    """
    if animation_type not in ("idle", "speaking"):
        animation_type = "idle"
    
    # Load any saved custom prompts for this agent
    custom_prompts = _load_agent_animation_prompts(agent_name)
    saved_idle_prompt = custom_prompts.get(expression)
    
    if animation_type == "idle":
        # Return saved custom prompt or default
        prompt = saved_idle_prompt or get_animation_prompt(expression, "idle")
    else:
        # For speaking, derive from saved idle prompt if available
        if saved_idle_prompt:
            prompt = _get_speaking_prompt_from_idle(saved_idle_prompt, expression)
        else:
            prompt = get_animation_prompt(expression, "speaking")
    
    return {
        "expression": expression,
        "animation_type": animation_type,
        "prompt": prompt,
        "has_custom_idle": saved_idle_prompt is not None,
    }


@router.delete(
    "/agents/{agent_name}/expressions/{expression}/animation",
    tags=["agents"],
)
async def delete_expression_animation(agent_name: str, expression: str, animation_type: str = "idle"):
    """Delete an animated GIF for an expression.

    Args:
        agent_name: Name of the agent.
        expression: Expression name.
        animation_type: Either 'idle' or 'speaking'.

    Returns:
        Deletion status.
    """
    from pathlib import Path
    
    # Check agent exists
    if agent_name not in list_agent_names():
        return JSONResponse(
            content={"message": f"Agent '{agent_name}' not found"},
            status_code=404,
        )
    
    if animation_type not in ("idle", "speaking"):
        animation_type = "idle"
    
    # Determine gif path based on animation type
    if animation_type == "speaking":
        gif_path = Path("agents") / agent_name / "expressions" / f"{expression}_speaking.gif"
    else:
        gif_path = Path("agents") / agent_name / "expressions" / f"{expression}.gif"
    
    if not gif_path.exists():
        return JSONResponse(
            content={"message": f"{animation_type.capitalize()} animation not found"},
            status_code=404,
        )
    
    try:
        gif_path.unlink()
        logger.info(f"Deleted {animation_type} animation: {gif_path}")
        return {
            "success": True,
            "expression": expression,
            "animation_type": animation_type,
            "deleted": str(gif_path),
        }
    except Exception as e:
        logger.error(f"Error deleting animation: {e}")
        return JSONResponse(
            content={"message": f"Error deleting animation: {str(e)}"},
            status_code=500,
        )


@router.get(
    "/agents/{agent_name}/expressions/{expression}_speaking.gif",
    tags=["agents"],
)
async def get_speaking_animation(agent_name: str, expression: str):
    """Serve a speaking animated GIF for an expression.

    Args:
        agent_name: Name of the agent.
        expression: Expression name (e.g., "neutral", "happy").

    Returns:
        The speaking animated GIF file.
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    gif_path = Path("agents") / agent_name / "expressions" / f"{expression}_speaking.gif"
    
    if not gif_path.exists():
        return JSONResponse(
            content={"message": "Speaking animation not found"},
            status_code=404,
        )
    
    return FileResponse(gif_path, media_type="image/gif")
