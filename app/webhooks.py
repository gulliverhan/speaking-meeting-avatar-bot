"""Webhook endpoints for ElevenLabs tool calls and Recall.ai transcription."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.models import ToolCallRequest, ToolCallResponse
from core.bot_state import BOT_STATES
from core.elevenlabs_client import ELEVENLABS_BRIDGES
from utils.logger import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post(
    "/tools/set_expression",
)
async def set_expression(request: Request):
    """Handle set_expression tool call from ElevenLabs agent.

    ElevenLabs webhook sends body params directly in the JSON body,
    not wrapped in a parameters object.

    When using browser SDK mode, we broadcast the expression change to all
    connected browsers since ElevenLabs doesn't provide a way to identify
    which specific conversation triggered the webhook.

    Args:
        request: The raw FastAPI request.

    Returns:
        Tool call response.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse set_expression body: {e}")
        return ToolCallResponse(success=False, error="Invalid JSON body")

    # Log the raw payload for debugging
    logger.info(f"set_expression webhook received: {body}")

    # ElevenLabs sends body params directly at top level
    expression = body.get("expression", "neutral")
    conversation_id = body.get("conversation_id")

    logger.info(f"Tool call: set_expression({expression}) for {conversation_id}")

    # Try to find the bot state by conversation_id
    bot_state = None
    client_id = None
    for cid, state in BOT_STATES.items():
        if state.elevenlabs_conversation_id == conversation_id:
            bot_state = state
            client_id = cid
            break

    # Send expression change to the browser
    from app.websockets import send_expression_to_browser, broadcast_expression_to_all_browsers
    
    if bot_state and client_id:
        # We found the specific bot - send to that browser
        bot_state.current_expression = expression
        await send_expression_to_browser(client_id, expression)
    else:
        # Can't identify specific bot - broadcast to all browsers
        # This is common in browser SDK mode where we don't track conversation_id
        logger.info(f"Broadcasting expression to all browsers (no specific bot found)")
        sent_count = await broadcast_expression_to_all_browsers(expression)
        
        # Also update all bot states (usually there's just one)
        for state in BOT_STATES.values():
            state.current_expression = expression
        
        if sent_count == 0:
            logger.warning("No browser connections to send expression to")

    return ToolCallResponse(
        success=True,
        result={"expression": expression},
    )


@router.post(
    "/tools/request_to_speak",
    response_model=ToolCallResponse,
)
async def request_to_speak(request: ToolCallRequest):
    """Handle request_to_speak tool call - shows hand-raised overlay.

    Args:
        request: The tool call request.

    Returns:
        Tool call response.
    """
    conversation_id = request.conversation_id

    logger.info(f"Tool call: request_to_speak() for {conversation_id}")

    # Find the bot state
    bot_state = None
    for state in BOT_STATES.values():
        if state.elevenlabs_conversation_id == conversation_id:
            bot_state = state
            break

    if not bot_state:
        return ToolCallResponse(
            success=False,
            error="Bot not found for this conversation",
        )

    # Set hand-raised flag
    bot_state.wants_to_speak = True

    # TODO: Send image with hand-raised overlay to Recall.ai

    return ToolCallResponse(
        success=True,
        result={"wants_to_speak": True},
    )


@router.post(
    "/tools/lower_hand",
    response_model=ToolCallResponse,
)
async def lower_hand(request: ToolCallRequest):
    """Handle lower_hand tool call - removes hand-raised overlay.

    Args:
        request: The tool call request.

    Returns:
        Tool call response.
    """
    conversation_id = request.conversation_id

    logger.info(f"Tool call: lower_hand() for {conversation_id}")

    # Find the bot state
    bot_state = None
    for state in BOT_STATES.values():
        if state.elevenlabs_conversation_id == conversation_id:
            bot_state = state
            break

    if not bot_state:
        return ToolCallResponse(
            success=False,
            error="Bot not found for this conversation",
        )

    # Clear hand-raised flag
    bot_state.wants_to_speak = False

    # TODO: Send image without overlay to Recall.ai

    return ToolCallResponse(
        success=True,
        result={"wants_to_speak": False},
    )


@router.post(
    "/tools/get_participants",
)
async def get_participants(request: Request):
    """Handle get_participants tool call from ElevenLabs agent.

    Returns the list of participants currently in the meeting.
    The response is spoken back to the user by the agent.

    Args:
        request: The raw FastAPI request.

    Returns:
        Tool call response with participant list.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse get_participants body: {e}")
        return ToolCallResponse(success=False, error="Invalid JSON body")

    logger.info(f"get_participants webhook received: {body}")

    # Since ElevenLabs doesn't send us a way to identify the bot,
    # we'll return participants from any active bot (usually just one)
    from core.recall_client import get_recall_client
    
    participants_info = []
    
    for client_id, state in BOT_STATES.items():
        if state.recall_bot_id:
            try:
                recall_client = get_recall_client()
                participants = await recall_client.get_meeting_participants(state.recall_bot_id)
                
                # If API returns participants, use those
                if participants:
                    for p in participants:
                        name = p.get("name", "Unknown")
                        is_host = p.get("is_host", False)
                        participants_info.append({
                            "name": name,
                            "is_host": is_host,
                        })
                else:
                    # Fallback 1: use participants detected via video frames
                    video_participants = state.extra.get("video_participants", {})
                    if video_participants:
                        logger.info(f"Using {len(video_participants)} video participants")
                        for pid, info in video_participants.items():
                            participants_info.append({
                                "name": info.get("name", "Unknown"),
                                "is_host": info.get("is_host", False),
                                "description": info.get("description", ""),
                            })
                    else:
                        # Fallback 2: use speakers we've seen from transcription
                        seen_speakers = state.extra.get("seen_speakers", {})
                        if seen_speakers:
                            logger.info(f"Using {len(seen_speakers)} seen speakers as participant fallback")
                            for name, info in seen_speakers.items():
                                participants_info.append({
                                    "name": name,
                                    "is_host": info.get("is_host", False),
                                })
                
                break  # Found an active bot
            except Exception as e:
                logger.error(f"Error getting participants: {e}")
    
    if participants_info:
        # Format for the agent to speak
        names = []
        for p in participants_info:
            name = p["name"]
            if p["is_host"]:
                name += " who is the host"
            names.append(name)
        
        result_text = f"Currently in the meeting: {', '.join(names)}"
        logger.info(f"Returning participants: {result_text}")
        
        return ToolCallResponse(
            success=True,
            result={"participants": participants_info, "summary": result_text},
        )
    else:
        return ToolCallResponse(
            success=True,
            result={"participants": [], "summary": "I couldn't find any participants in the meeting."},
        )


# Cache for get_meeting_context results (client_id -> (timestamp, result_text))
# Prevents rapid repeated calls from burning through Replicate credits
MEETING_CONTEXT_CACHE: dict[str, tuple[float, str]] = {}
CONTEXT_CACHE_TTL = 30  # seconds - return cached results if called within this time


@router.api_route(
    "/tools/get_meeting_context",
    methods=["GET", "POST"],
)
async def get_meeting_context(request: Request):
    """Handle get_meeting_context tool call from ElevenLabs agent.

    Returns visual context about participants - what they look like, their background, etc.
    The agent should call this when joining and periodically to get updates.
    
    Results are cached for CONTEXT_CACHE_TTL seconds to prevent rapid repeated calls
    from burning through Replicate credits.

    Args:
        request: The raw FastAPI request.

    Returns:
        Tool call response with visual context.
    """
    import asyncio
    import base64
    import time
    from pathlib import Path
    from app.websockets import analyze_participant_image
    
    # Handle both GET (no body) and POST (with body)
    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            pass  # No body is fine
    
    logger.info(f"get_meeting_context webhook received ({request.method})")

    # Find any active bot and get visual context from video participants
    for client_id, state in BOT_STATES.items():
        video_participants = state.extra.get("video_participants", {})
        
        if video_participants:
            current_time = time.time()
            
            # Check cache first - return cached result if recent enough
            if client_id in MEETING_CONTEXT_CACHE:
                cached_time, cached_result = MEETING_CONTEXT_CACHE[client_id]
                cache_age = current_time - cached_time
                if cache_age < CONTEXT_CACHE_TTL:
                    logger.info(f"Returning cached context (age: {cache_age:.1f}s)")
                    return ToolCallResponse(
                        success=True,
                        result={"context": cached_result},
                    )
            
            # Re-analyze ALL streams in parallel
            async def analyze_stream(stream_key: str, info: dict) -> tuple[str, str, bool]:
                """Analyze a single stream and return (stream_key, description, is_screen_share)."""
                name = info.get("name", "Someone")
                frame_path = info.get("frame_path")
                is_screen_share = info.get("is_screen_share", False)
                
                if frame_path:
                    try:
                        frame_file = Path(frame_path)
                        if frame_file.exists():
                            with open(frame_file, "rb") as f:
                                image_data = f.read()
                            image_b64 = base64.b64encode(image_data).decode()
                            
                            # Re-analyze with AI vision
                            new_description = await analyze_participant_image(
                                name, image_b64, frame_path
                            )
                            
                            if new_description:
                                # Update stored description
                                state.extra["video_participants"][stream_key]["description"] = new_description
                                return (stream_key, new_description, is_screen_share)
                    except Exception as e:
                        logger.error(f"Failed to analyze {stream_key}: {e}")
                
                # Fall back to existing description
                return (stream_key, info.get("description", ""), is_screen_share)
            
            # Launch all analyses in parallel
            logger.info(f"Re-analyzing {len(video_participants)} streams in parallel...")
            tasks = [
                analyze_stream(stream_key, info) 
                for stream_key, info in video_participants.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            people_parts = []
            screen_share_parts = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis task failed: {result}")
                    continue
                    
                stream_key, description, is_screen_share = result
                info = video_participants.get(stream_key, {})
                name = info.get("name", "Someone")
                is_host = info.get("is_host", False)
                
                if is_screen_share:
                    if description:
                        screen_share_parts.append(f"{name} is sharing their screen: {description}")
                    else:
                        screen_share_parts.append(f"{name} is sharing their screen")
                else:
                    if description:
                        host_str = " (they're the host)" if is_host else ""
                        people_parts.append(f"{name}{host_str}: {description}")
                    else:
                        host_str = " who is the host" if is_host else ""
                        people_parts.append(f"{name}{host_str} is on the call")
            
            # Build the result
            result_parts = []
            if people_parts:
                result_parts.append("PARTICIPANTS:\n" + "\n".join(f"- {p}" for p in people_parts))
            if screen_share_parts:
                result_parts.append("SCREEN SHARES:\n" + "\n".join(f"- {s}" for s in screen_share_parts))
            
            result_text = "\n\n".join(result_parts) if result_parts else "No participants detected yet"
            
            # Cache the result
            MEETING_CONTEXT_CACHE[client_id] = (current_time, result_text)
            logger.info(f"Returning fresh meeting context (cached for {CONTEXT_CACHE_TTL}s): {result_text[:100]}...")
            
            return ToolCallResponse(
                success=True,
                result={"context": result_text},
            )
    
    # No visual context available yet
    return ToolCallResponse(
        success=True,
        result={"context": "I can't see anyone in the meeting yet - the video might still be loading."},
    )


@router.post(
    "/tools/generate_image",
    response_model=ToolCallResponse,
)
async def generate_image(request: ToolCallRequest):
    """Handle generate_image tool call - generates image via Replicate.

    Args:
        request: The tool call request.

    Returns:
        Tool call response.
    """
    prompt = request.parameters.get("prompt", "")
    conversation_id = request.conversation_id

    logger.info(f"Tool call: generate_image({prompt[:50]}...) for {conversation_id}")

    if not prompt:
        return ToolCallResponse(
            success=False,
            error="Prompt is required",
        )

    # Find the bot state
    bot_state = None
    for state in BOT_STATES.values():
        if state.elevenlabs_conversation_id == conversation_id:
            bot_state = state
            break

    if not bot_state:
        return ToolCallResponse(
            success=False,
            error="Bot not found for this conversation",
        )

    # TODO: Call Replicate to generate image
    # TODO: Send generated image to Recall.ai output_media

    return ToolCallResponse(
        success=True,
        result={"status": "generating", "prompt": prompt},
    )


# =============================================================================
# Recall.ai Transcription Webhook
# =============================================================================

@router.post("/transcription/{client_id}")
async def receive_transcription(client_id: str, request: Request):
    """Receive real-time transcription events from Recall.ai.

    This webhook is called when using Recall.ai transcription (speaker-aware mode).
    We inject the speaker info into the ElevenLabs conversation context.

    Payload format from Recall.ai (see https://docs.recall.ai/docs/bot-real-time-transcription):
    {
        "event": "transcript.data",
        "data": {
            "data": {
                "words": [{"text": "hello", ...}],
                "participant": {"id": 1, "name": "John", ...}
            },
            "bot": {"id": "...", "metadata": {}},
            ...
        }
    }

    Args:
        client_id: The bot's client ID.
        request: The incoming request with transcription data.

    Returns:
        Acknowledgment response.
    """
    try:
        payload: dict[str, Any] = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse transcription webhook payload: {e}")
        return {"status": "error", "message": "Invalid JSON"}

    # Get the bot state
    bot_state = BOT_STATES.get(client_id)
    if not bot_state:
        logger.warning(f"Transcription webhook: No bot found for client {client_id}")
        return {"status": "ignored", "message": "Bot not found"}

    # Check if speaker context is enabled
    if not bot_state.speaker_context:
        return {"status": "ok", "message": "Speaker context disabled"}

    # Check event type - only process final transcript data
    event_type = payload.get("event")
    if event_type == "transcript.partial_data":
        return {"status": "ok", "message": "Ignoring partial transcript"}

    # Extract transcription data from Recall.ai webhook
    # See: https://docs.recall.ai/docs/bot-real-time-transcription
    data = payload.get("data", {}).get("data", {})
    
    # Get words and combine into transcript
    words = data.get("words", [])
    transcript = " ".join(word.get("text", "") for word in words).strip()
    
    # Get speaker info
    participant = data.get("participant", {})
    speaker = participant.get("name") or f"Participant {participant.get('id', 'Unknown')}"
    participant_id = participant.get("id")
    is_host = participant.get("is_host", False)

    if not transcript:
        return {"status": "ok", "message": "No transcript text"}

    logger.info(f"[{client_id}] 🎤 {speaker} is speaking")
    
    # Track this speaker - use transcription as source of participant info
    # since meeting_participants may be empty on some platforms
    if "seen_speakers" not in bot_state.extra:
        bot_state.extra["seen_speakers"] = {}
    
    if speaker and speaker not in bot_state.extra["seen_speakers"]:
        bot_state.extra["seen_speakers"][speaker] = {
            "id": participant_id,
            "name": speaker,
            "is_host": is_host,
            "first_seen": True,
        }
        logger.info(f"[{client_id}] New speaker detected: {speaker}")
        
        # Inject context about new speaker joining (first time we hear them)
        from app.websockets import send_context_to_browser
        join_context = f"[{speaker} is now in the meeting]"
        await send_context_to_browser(client_id, join_context, speaker="System")

    # Build simple speaker context message for ElevenLabs
    # We don't send the transcript since ElevenLabs is already transcribing
    # Just let the AI know who is currently speaking
    context_message = f"[{speaker} is speaking]"

    # Try to send context to ElevenLabs
    # Method 1: Server-side bridge (if using server-based audio routing)
    bridge = ELEVENLABS_BRIDGES.get(client_id)
    if bridge:
        try:
            await bridge.elevenlabs_client.send_contextual_update(context_message)
            logger.debug(f"Injected speaker context via server bridge for {client_id}")
            return {"status": "ok", "speaker": speaker, "transcript_length": len(transcript)}
        except Exception as e:
            logger.error(f"Failed to inject via server bridge: {e}")

    # Method 2: Send to browser WebSocket (if using webpage-based output)
    from app.websockets import send_context_to_browser
    
    sent_to_browser = await send_context_to_browser(client_id, context_message, speaker)
    if sent_to_browser:
        logger.debug(f"Sent speaker context to browser for {client_id}")
        return {"status": "ok", "speaker": speaker, "transcript_length": len(transcript), "via": "browser"}
    
    # Neither method worked
    logger.warning(f"Could not inject context for {client_id} - no bridge or browser connection")
    return {"status": "warning", "message": "No active connection to inject context"}
