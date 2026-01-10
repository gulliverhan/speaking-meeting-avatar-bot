"""WebSocket routes for handling Recall.ai audio streaming and browser communication."""

import asyncio
import base64
import io
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydub import AudioSegment

from core.agent_manager import get_agent_config
from core.bot_state import BOT_STATES
from core.elevenlabs_client import (
    ELEVENLABS_BRIDGES,
    ElevenLabsBridge,
    create_bridge,
    remove_bridge,
)
from core.recall_client import get_recall_client
from utils.logger import logger
from utils.prompts import format_prompt

router = APIRouter()

# Store browser WebSocket connections by client_id
# These are used to send transcription context from server to browser
BROWSER_CONNECTIONS: dict[str, WebSocket] = {}

# Audio buffer settings for PCM to MP3 conversion
# ElevenLabs sends 16-bit PCM at 16kHz mono
ELEVENLABS_SAMPLE_RATE = 16000
ELEVENLABS_CHANNELS = 1
ELEVENLABS_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# Buffer size before sending (in samples) - ~500ms of audio
AUDIO_BUFFER_SAMPLES = ELEVENLABS_SAMPLE_RATE // 2


def pcm_to_mp3_base64(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> str:
    """Convert raw PCM audio to MP3 and return as base64.
    
    Args:
        pcm_data: Raw PCM audio bytes (16-bit).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        sample_width: Bytes per sample (2 for 16-bit).
        
    Returns:
        Base64 encoded MP3 audio.
    """
    try:
        # Create AudioSegment from raw PCM data
        audio = AudioSegment(
            data=pcm_data,
            sample_width=sample_width,
            frame_rate=sample_rate,
            channels=channels,
        )
        
        # Export to MP3 in memory
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="64k")
        mp3_buffer.seek(0)
        
        # Return as base64
        return base64.b64encode(mp3_buffer.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting PCM to MP3: {e}")
        return ""


async def handle_tool_call(
    bot_id: str,
    tool_name: str,
    parameters: dict,
) -> None:
    """Handle tool calls from the ElevenLabs agent.

    Args:
        bot_id: The bot's client ID.
        tool_name: Name of the tool being called.
        parameters: Tool parameters.
    """
    bot_state = BOT_STATES.get(bot_id)
    if not bot_state:
        logger.warning(f"No bot state for tool call: {bot_id}")
        return

    logger.info(f"Tool call for {bot_id}: {tool_name}({parameters})")

    if tool_name == "set_expression":
        expression = parameters.get("expression", "neutral")
        bot_state.current_expression = expression
        logger.info(f"Expression set to: {expression}")
        # TODO: Send updated image to Recall.ai

    elif tool_name == "request_to_speak":
        bot_state.wants_to_speak = True
        logger.info("Hand raised - wants to speak")
        # TODO: Send image with hand overlay to Recall.ai

    elif tool_name == "lower_hand":
        bot_state.wants_to_speak = False
        logger.info("Hand lowered")
        # TODO: Send image without overlay to Recall.ai

    elif tool_name == "generate_image":
        prompt = parameters.get("prompt", "")
        logger.info(f"Generate image: {prompt}")
        # TODO: Call Replicate and send to Recall.ai


@router.websocket("/ws/browser/{client_id}")
async def browser_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections from the browser (index.html).

    This endpoint allows the server to send transcription context to the
    browser, which can then inject it into the ElevenLabs conversation.

    Args:
        websocket: The WebSocket connection.
        client_id: The bot's client ID.
    """
    await websocket.accept()
    logger.info(f"Browser WebSocket connected: {client_id}")

    # Store the connection for sending context updates
    BROWSER_CONNECTIONS[client_id] = websocket

    try:
        # Keep connection alive and handle any messages from browser
        while True:
            try:
                message = await websocket.receive()
                if "text" in message:
                    # Browser might send status updates
                    logger.debug(f"Browser message from {client_id}: {message['text'][:100]}")
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    break
                raise

    except WebSocketDisconnect:
        logger.info(f"Browser WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Browser WebSocket error for {client_id}: {e}")
    finally:
        BROWSER_CONNECTIONS.pop(client_id, None)
        logger.info(f"Browser WebSocket cleanup complete for {client_id}")


async def send_context_to_browser(client_id: str, context: str, speaker: str = None) -> bool:
    """Send transcription context to the browser for injection into ElevenLabs.

    Args:
        client_id: The bot's client ID.
        context: The context message to send.
        speaker: Optional speaker name.

    Returns:
        True if sent successfully, False otherwise.
    """
    ws = BROWSER_CONNECTIONS.get(client_id)
    if not ws:
        logger.debug(f"No browser connection for {client_id}")
        return False

    try:
        import json
        await ws.send_text(json.dumps({
            "type": "context_update",
            "context": context,
            "speaker": speaker
        }))
        return True
    except Exception as e:
        logger.error(f"Error sending context to browser {client_id}: {e}")
        return False


async def send_user_message_to_browser(client_id: str, message: str) -> bool:
    """Send a message to the browser that will be treated as user input.
    
    Unlike context_update (background info), this will make the agent respond.

    Args:
        client_id: The bot's client ID.
        message: The message to send as if from the user.

    Returns:
        True if sent successfully, False otherwise.
    """
    ws = BROWSER_CONNECTIONS.get(client_id)
    if not ws:
        logger.debug(f"No browser connection for {client_id}")
        return False

    try:
        import json
        await ws.send_text(json.dumps({
            "type": "user_message",
            "message": message
        }))
        return True
    except Exception as e:
        logger.error(f"Error sending user message to browser {client_id}: {e}")
        return False


async def send_expression_to_browser(client_id: str, expression: str) -> bool:
    """Send expression change to the browser to update the avatar.

    Args:
        client_id: The bot's client ID.
        expression: The expression name (e.g., "happy", "thinking").

    Returns:
        True if sent successfully, False otherwise.
    """
    ws = BROWSER_CONNECTIONS.get(client_id)
    if not ws:
        logger.debug(f"No browser connection for {client_id}")
        return False

    try:
        import json
        await ws.send_text(json.dumps({
            "type": "expression_change",
            "expression": expression
        }))
        logger.info(f"Sent expression '{expression}' to browser {client_id}")
        return True
    except Exception as e:
        logger.error(f"Error sending expression to browser {client_id}: {e}")
        return False


async def broadcast_expression_to_all_browsers(expression: str) -> int:
    """Broadcast expression change to all connected browsers.
    
    This is used when we can't identify which specific bot the expression
    change is for (e.g., when ElevenLabs webhook doesn't include conversation_id).

    Args:
        expression: The expression name.

    Returns:
        Number of browsers the message was sent to.
    """
    import json
    message = json.dumps({
        "type": "expression_change",
        "expression": expression
    })
    
    sent_count = 0
    for client_id, ws in list(BROWSER_CONNECTIONS.items()):
        try:
            await ws.send_text(message)
            logger.info(f"Broadcast expression '{expression}' to browser {client_id}")
            sent_count += 1
        except Exception as e:
            logger.error(f"Error broadcasting to {client_id}: {e}")
    
    return sent_count


@router.websocket("/ws/audio/{client_id}")
async def audio_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket for receiving meeting audio from Recall.ai.

    This endpoint:
    1. Receives meeting audio from Recall.ai via WebSocket
    2. Forwards it to ElevenLabs for processing
    3. Sends ElevenLabs responses back via Output Audio API (separate channel)

    This prevents echo because input and output are completely separate!

    Args:
        websocket: The WebSocket connection.
        client_id: The bot's client ID.
    """
    await websocket.accept()
    logger.info(f"Audio WebSocket connected: {client_id}")

    # Get bot state
    bot_state = BOT_STATES.get(client_id)
    if not bot_state:
        logger.error(f"No bot state found for client: {client_id}")
        await websocket.close(code=1008, reason="Bot not found")
        return

    bridge: Optional[ElevenLabsBridge] = None
    audio_buffer: list[bytes] = []
    buffer_lock = asyncio.Lock()
    is_speaking = False

    # Get Recall.ai client for sending audio back
    recall_client = get_recall_client()

    async def flush_audio_buffer():
        """Convert buffered PCM to MP3 and send to Recall.ai."""
        nonlocal audio_buffer, is_speaking
        
        async with buffer_lock:
            if not audio_buffer:
                return
                
            # Combine all buffered chunks
            combined_pcm = b"".join(audio_buffer)
            audio_buffer = []
        
        if len(combined_pcm) < 1000:  # Skip very small chunks
            return
            
        # Convert to MP3 and send
        mp3_base64 = pcm_to_mp3_base64(
            combined_pcm,
            sample_rate=ELEVENLABS_SAMPLE_RATE,
            channels=ELEVENLABS_CHANNELS,
            sample_width=ELEVENLABS_SAMPLE_WIDTH,
        )
        
        if mp3_base64 and bot_state.recall_bot_id:
            success = await recall_client.send_audio(bot_state.recall_bot_id, mp3_base64)
            if success:
                logger.debug(f"Sent {len(combined_pcm)} bytes of audio to meeting")
            else:
                logger.warning(f"Failed to send audio to meeting for {client_id}")

    async def on_audio_from_agent(audio_data: bytes) -> None:
        """Handle audio received from ElevenLabs - buffer and send via Output Audio API."""
        nonlocal is_speaking
        is_speaking = True
        
        async with buffer_lock:
            audio_buffer.append(audio_data)
            
            # Check if buffer is large enough to send
            total_size = sum(len(chunk) for chunk in audio_buffer)
            if total_size >= AUDIO_BUFFER_SAMPLES * ELEVENLABS_SAMPLE_WIDTH:
                # Schedule flush (don't await to avoid blocking)
                asyncio.create_task(flush_audio_buffer())

    async def on_agent_done_speaking():
        """Called when ElevenLabs finishes speaking - flush remaining buffer."""
        nonlocal is_speaking
        is_speaking = False
        await flush_audio_buffer()

    try:
        # Load agent config to get ElevenLabs agent ID
        agent_config = get_agent_config(bot_state.agent_name)
        if not agent_config:
            logger.error(f"Agent config not found: {bot_state.agent_name}")
            await websocket.close(code=1008, reason="Agent config not found")
            return

        # Get ElevenLabs agent ID from config or env
        elevenlabs_agent_id = agent_config.get("elevenlabs", {}).get(
            "agent_id"
        ) or os.getenv("ELEVENLABS_AGENT_ID")

        if not elevenlabs_agent_id:
            logger.error("No ElevenLabs agent ID configured")
            await websocket.close(code=1008, reason="No ElevenLabs agent ID")
            return

        # Create ElevenLabs bridge with callbacks
        async def on_tool_call(tool_name: str, params: dict):
            await handle_tool_call(client_id, tool_name, params)

        bridge = await create_bridge(
            client_id=client_id,
            agent_id=elevenlabs_agent_id,
            on_tool_call=on_tool_call,
        )

        # Store conversation ID in bot state
        if bridge.conversation_id:
            bot_state.elevenlabs_conversation_id = bridge.conversation_id

        logger.info(f"ElevenLabs bridge created for {client_id}")

        # Set up callback to handle audio from ElevenLabs
        # This routes to Output Audio API instead of the WebSocket
        bridge.set_meeting_audio_callback(on_audio_from_agent)

        # Process incoming audio from Recall.ai
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info(f"Audio WebSocket closed by client: {client_id}")
                    break
                raise

            if "bytes" in message:
                audio_data = message["bytes"]
                
                # Don't forward to ElevenLabs while we're speaking
                # This prevents the bot from hearing its own voice
                if not is_speaking:
                    await bridge.send_audio_to_agent(audio_data)

            elif "text" in message:
                text_data = message["text"]
                logger.debug(f"Received text from {client_id}: {text_data[:100]}")

    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected: {client_id}")
    except ConnectionError as e:
        logger.error(f"Failed to create ElevenLabs bridge: {e}")
    except Exception as e:
        logger.error(f"Audio WebSocket error for {client_id}: {e}")
    finally:
        # Flush any remaining audio
        await flush_audio_buffer()
        
        # Clean up ElevenLabs bridge
        if bridge or client_id in ELEVENLABS_BRIDGES:
            await remove_bridge(client_id)
            logger.info(f"ElevenLabs bridge removed for {client_id}")

        logger.info(f"Audio WebSocket cleanup complete for {client_id}")


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Legacy WebSocket endpoint - redirects to audio endpoint.
    
    Kept for backwards compatibility.
    """
    # Just forward to the audio endpoint
    await audio_websocket_endpoint(websocket, client_id)


# Store participant video frames for analysis
# Key: (client_id, participant_id) -> {"name": str, "frame": base64, "analyzed": bool, "description": str}
PARTICIPANT_FRAMES: dict[tuple[str, int], dict] = {}


async def analyze_participant_image(participant_name: str, image_base64: str, image_path: str = None) -> str:
    """Analyze a participant's video frame using Replicate's Gemini 2.5 Flash.
    
    Args:
        participant_name: The participant's name.
        image_base64: Base64-encoded PNG image.
        image_path: Path to saved image file (used to serve via ngrok).
        
    Returns:
        A description of what's in the image.
    """
    import os
    import httpx
    
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        logger.warning("REPLICATE_API_TOKEN not set, skipping image analysis")
        return ""
    
    # Get public URL for the image (serve via ngrok)
    from utils.ngrok import get_public_url
    ngrok_url = get_public_url()
    
    if not ngrok_url or not image_path:
        logger.warning("Cannot analyze image: no ngrok URL or image path")
        return ""
    
    # Convert local path to public URL
    # image_path is like "participant_frames/{client_id}/{name}_{id}.png"
    image_url = f"{ngrok_url}/{image_path}"
    logger.info(f"Analyzing image at: {image_url}")
    
    max_retries = 3
    retry_delay = 8  # seconds - Replicate suggests ~7s for rate limit reset
    
    try:
        async with httpx.AsyncClient() as client:
            # Start prediction using Gemini 2.5 Flash (with retry for rate limits)
            response = None
            for attempt in range(max_retries):
                response = await client.post(
                    "https://api.replicate.com/v1/models/google/gemini-2.5-flash/predictions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": {
                            "prompt": format_prompt("participant_analysis", participant_name=participant_name),
                            "images": [image_url],
                            "top_p": 0.95,
                            "temperature": 1,
                            "max_output_tokens": 1000,
                        }
                    },
                    timeout=30.0,
                )
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    logger.warning(f"Replicate rate limited, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                break  # Success or non-retryable error
            
            if response.status_code != 201:
                logger.error(f"Replicate API error: {response.status_code} - {response.text}")
                return ""
            
            prediction = response.json()
            prediction_url = prediction.get("urls", {}).get("get")
            
            if not prediction_url:
                logger.error("No prediction URL returned")
                return ""
            
            # Poll for completion
            for _ in range(30):  # Max 30 seconds
                await asyncio.sleep(1)
                
                poll_response = await client.get(
                    prediction_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0,
                )
                
                if poll_response.status_code != 200:
                    continue
                    
                result = poll_response.json()
                status = result.get("status")
                
                if status == "succeeded":
                    output = result.get("output", "")
                    logger.debug(f"Raw vision output type: {type(output)}, value: {output}")
                    
                    # Output might be a string or list of tokens
                    if isinstance(output, list):
                        description = "".join(output)
                    else:
                        description = str(output)
                    
                    logger.info(f"Analyzed {participant_name}: {description}")
                    return description.strip()
                elif status == "failed":
                    logger.error(f"Prediction failed: {result.get('error')}")
                    return ""
            
            logger.warning("Prediction timed out")
            return ""
                
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return ""


# How often to save frames (seconds) - balance between freshness and disk I/O
FRAME_SAVE_INTERVAL = 30

@router.websocket("/ws/video/{client_id}")
async def video_websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket for receiving per-participant video frames from Recall.ai.
    
    This endpoint receives video_separate_png.data events which include:
    - PNG frame of each participant
    - Participant info (name, id, is_host, email)
    
    We use this to:
    1. Track participants in meetings (even on Google Meet!)
    2. Analyze frames with AI Vision for context (on first appearance + on-demand refresh)
    
    Frames are saved every FRAME_SAVE_INTERVAL seconds (not every frame) to reduce disk I/O.
    
    Args:
        websocket: The WebSocket connection from Recall.ai.
        client_id: The bot's client ID.
    """
    import time
    
    await websocket.accept()
    logger.info(f"Video WebSocket connected: {client_id}")
    
    # Track which participants we've seen and analyzed (for initial analysis only)
    seen_participants: set[str] = set()
    # Track last save time per stream (stream_key -> timestamp)
    last_save_times: dict[str, float] = {}
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info(f"Video WebSocket closed: {client_id}")
                    break
                raise
            
            if "text" in message:
                try:
                    import json
                    data = json.loads(message["text"])
                    event_type = data.get("event", "unknown")
                    
                    # Handle video_separate_png.data event
                    if event_type == "video_separate_png.data":
                        frame_data = data.get("data", {}).get("data", {})
                        participant = frame_data.get("participant", {})
                        
                        participant_id = participant.get("id")
                        participant_name = participant.get("name") or f"Participant {participant_id}"
                        is_host = participant.get("is_host", False)
                        image_buffer = frame_data.get("buffer", "")  # base64 PNG
                        frame_type = frame_data.get("type", "unknown")  # "webcam" or "screenshare"
                        
                        # Use (participant_id, frame_type) as unique key to capture both camera AND screen share
                        stream_key = f"{participant_id}_{frame_type}"
                        is_new_stream = stream_key not in seen_participants
                        
                        # Only log details for NEW streams (not every frame)
                        if is_new_stream:
                            logger.info(f"=== NEW VIDEO STREAM: {participant_name} ({frame_type}) ===")
                            seen_participants.add(stream_key)
                        
                        # Check if we should save this frame (first frame OR 30s since last save)
                        current_time = time.time()
                        last_save = last_save_times.get(stream_key, 0)
                        should_save = is_new_stream or (current_time - last_save >= FRAME_SAVE_INTERVAL)
                        
                        saved_image_path = None
                        if should_save and image_buffer and participant_id:
                            import base64
                            from pathlib import Path
                            
                            # Create output directory
                            output_dir = Path("participant_frames") / client_id
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save the PNG - include frame_type to differentiate camera vs screen share
                            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in participant_name)
                            safe_type = "".join(c if c.isalnum() else "_" for c in str(frame_type))
                            filename = output_dir / f"{safe_name}_{participant_id}_{safe_type}.png"
                            
                            try:
                                image_bytes = base64.b64decode(image_buffer)
                                with open(filename, "wb") as f:
                                    f.write(image_bytes)
                                saved_image_path = str(filename)
                                last_save_times[stream_key] = current_time
                                
                                if is_new_stream:
                                    logger.info(f"Saved initial frame: {filename}")
                                else:
                                    logger.debug(f"Saved periodic frame: {filename}")
                            except Exception as e:
                                logger.error(f"Failed to save frame: {e}")
                        
                        # Store/update participant info in bot state
                        bot_state = BOT_STATES.get(client_id)
                        is_screen_share = frame_type == "screenshare"
                        
                        if bot_state:
                            if "video_participants" not in bot_state.extra:
                                bot_state.extra["video_participants"] = {}
                            
                            # Get existing info to preserve frame_path if we didn't save
                            existing_info = bot_state.extra["video_participants"].get(stream_key, {})
                            
                            # Only update frame_path if we actually saved a new frame
                            frame_path = saved_image_path if saved_image_path else existing_info.get("frame_path")
                            
                            bot_state.extra["video_participants"][stream_key] = {
                                "name": participant_name,
                                "is_host": is_host,
                                "email": participant.get("email"),
                                "stream_type": frame_type,  # "webcam" or "screenshare"
                                "is_screen_share": is_screen_share,
                                "frame_path": frame_path,  # Keep existing path if we didn't save
                                "description": existing_info.get("description", ""),  # Keep old until refreshed
                            }
                            
                            # Only analyze FIRST frame of each stream (on-demand refresh for changes)
                            if is_new_stream and saved_image_path:
                                description = await analyze_participant_image(
                                    participant_name, image_buffer, saved_image_path
                                )
                                
                                if description:
                                    bot_state.extra["video_participants"][stream_key]["description"] = description
                                    
                                    # Inject as context to ElevenLabs
                                    if is_screen_share:
                                        context = f"[Screen share from {participant_name}: {description}]"
                                    else:
                                        context = f"[Visual context: {participant_name} - {description}]"
                                    await send_context_to_browser(client_id, context, speaker="System")
                            
                            # Send notification for new streams
                            if is_new_stream:
                                if not is_screen_share:
                                    join_context = f"[{participant_name} is in the meeting" + (" as host" if is_host else "") + "]"
                                    await send_context_to_browser(client_id, join_context, speaker="System")
                                else:
                                    # Notify about screen sharing
                                    share_context = f"[{participant_name} started sharing their screen]"
                                    await send_context_to_browser(client_id, share_context, speaker="System")
                    else:
                        # Log non-video events normally
                        logger.info(f"Video WebSocket event: {event_type}")
                    
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON message on video WebSocket: {message['text'][:100]}")
                except Exception as e:
                    logger.error(f"Error processing video message: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"Video WebSocket disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Video WebSocket error for {client_id}: {e}")
    finally:
        # Clean up participant frames
        keys_to_remove = [k for k in PARTICIPANT_FRAMES if k[0] == client_id]
        for key in keys_to_remove:
            PARTICIPANT_FRAMES.pop(key, None)
        logger.info(f"Video WebSocket cleanup complete for {client_id}")
