"""Image generation utilities for avatar creation.

Supports Replicate models:
- google/nano-banana-pro (highest quality, supports image input)
- google/nano-banana (good quality, supports image input)
- prunaai/p-image (cheap and fast)
- bytedance/seedance-1-pro-fast (video/animation generation)

All configured for small image sizes suitable for video conferencing avatars.
"""

import asyncio
import os
import subprocess
import tempfile
import httpx
from pathlib import Path
from typing import Any, Optional

from utils.logger import logger
from utils.prompts import format_prompt, get_expression_modifier, load_yaml_prompts


# Available Replicate models with their configurations
REPLICATE_MODELS = {
    "nano-banana-pro": {
        "model": "google/nano-banana-pro",
        "description": "Highest quality, Google's latest, supports reference images",
        "speed": "slow",
        "cost": "high",
        "supports_image_input": True,
    },
    "nano-banana": {
        "model": "google/nano-banana",
        "description": "Good quality, faster, supports reference images",
        "speed": "medium",
        "cost": "medium",
        "supports_image_input": True,
    },
    "p-image": {
        "model": "prunaai/p-image",
        "description": "Cheap and fast",
        "speed": "fast",
        "cost": "low",
        "supports_image_input": False,
    },
}

DEFAULT_MODEL = "p-image"  # Fast and cheap by default


# Expression modifiers are now loaded from prompts/expression_modifiers.yaml


def get_available_models() -> list[dict[str, Any]]:
    """Get list of available image generation models.

    Returns:
        List of model info dicts.
    """
    return [
        {"id": key, **value}
        for key, value in REPLICATE_MODELS.items()
    ]


async def generate_avatar_image(
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    reference_image_url: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a single avatar image.

    Args:
        prompt: Description of the avatar character.
        model_id: Model to use (nano-banana-pro, nano-banana, p-image).
        reference_image_url: Optional URL of a reference image to base the avatar on.

    Returns:
        Dict with success status and image_url or error.
    """
    # Load and format prompt templates
    if reference_image_url:
        enhanced_prompt = format_prompt("avatar_from_reference", prompt=prompt)
    else:
        enhanced_prompt = format_prompt("avatar_from_scratch", prompt=prompt)
    
    # Fallback if prompt files are missing
    if not enhanced_prompt:
        enhanced_prompt = f"Professional headshot portrait photo of {prompt}"

    return await _generate_replicate(enhanced_prompt, model_id, reference_image_url)


async def generate_expression_pack(
    base_prompt: str,
    base_image_url: str,
    expressions: list[str],
    output_dir: Path,
    model_id: str = DEFAULT_MODEL,
    reference_image_url: Optional[str] = None,
    background_instruction: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a pack of expression images based on a character description.

    Args:
        base_prompt: Original character description.
        base_image_url: URL of the approved base image (for reference).
        expressions: List of expressions to generate.
        output_dir: Directory to save generated images.
        model_id: Replicate model to use.
        reference_image_url: Optional reference image URL.
        background_instruction: Background setting - None for default, "match_base" to match
                               the base avatar, or a custom description string.

    Returns:
        Dict with success status and list of generated expressions.
    """
    generated = []
    errors = []
    
    # Build background part of prompt
    if background_instruction == "match_base":
        bg_prompt = "Keep the EXACT same background as the reference image."
    elif background_instruction:
        bg_prompt = f"Background: {background_instruction}"
    else:
        bg_prompt = "Clean neutral background."

    for expression in expressions:
        modifier = get_expression_modifier(expression)
        
        # Build expression-specific prompt using templates
        if reference_image_url:
            expr_prompt = format_prompt("expression_from_reference", modifier=modifier, background_instruction=bg_prompt)
        else:
            expr_prompt = format_prompt("expression_from_scratch", base_prompt=base_prompt, modifier=modifier, background_instruction=bg_prompt)
        
        # Fallback if prompt files are missing
        if not expr_prompt:
            expr_prompt = f"Professional headshot portrait of {base_prompt}, {modifier}. {bg_prompt}"

        logger.info(f"Generating expression: {expression}")
        
        try:
            result = await _generate_replicate(expr_prompt, model_id, reference_image_url)
            
            if result.get("success"):
                # Download and save the image
                image_url = result["image_url"]
                image_path = output_dir / f"{expression}.png"
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    image_path.write_bytes(response.content)
                
                generated.append({
                    "expression": expression,
                    "image_path": str(image_path),
                    "image_url": image_url,
                })
                logger.info(f"Saved {expression} to {image_path}")
            else:
                errors.append({
                    "expression": expression,
                    "error": result.get("error", "Unknown error"),
                })
                logger.error(f"Failed to generate {expression}: {result.get('error')}")
                
        except Exception as e:
            errors.append({
                "expression": expression,
                "error": str(e),
            })
            logger.error(f"Error generating {expression}: {e}")

    return {
        "success": len(generated) > 0,
        "expressions": generated,
        "errors": errors if errors else None,
        "total": len(expressions),
        "generated": len(generated),
        "failed": len(errors),
    }


async def _generate_replicate(
    prompt: str, 
    model_id: str = DEFAULT_MODEL,
    reference_image_url: Optional[str] = None,
) -> dict[str, Any]:
    """Generate image using Replicate.

    Args:
        prompt: Image generation prompt.
        model_id: Model key (nano-banana-pro, nano-banana, p-image).
        reference_image_url: Optional reference image URL.

    Returns:
        Dict with success status and image_url or error.
    """
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        return {"success": False, "error": "REPLICATE_API_TOKEN not set in environment"}

    # Get model configuration
    model_config = REPLICATE_MODELS.get(model_id)
    if not model_config:
        return {"success": False, "error": f"Unknown model: {model_id}. Available: {list(REPLICATE_MODELS.keys())}"}
    
    model_name = model_config["model"]
    
    # Check if trying to use reference image with unsupported model
    if reference_image_url and not model_config.get("supports_image_input"):
        logger.warning(f"Model {model_id} doesn't support image input, ignoring reference image")
        reference_image_url = None
    
    logger.info(f"Using Replicate model: {model_name}" + (f" with reference image" if reference_image_url else ""))

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            # Build input based on model - optimized for small avatar images
            if model_id == "nano-banana-pro":
                model_input = {
                    "prompt": prompt,
                    "resolution": "1K",  # Smallest available (1024px) - good for avatars
                    "aspect_ratio": "1:1",  # Square for avatars
                    "output_format": "png",
                    "safety_filter_level": "block_only_high",
                }
                if reference_image_url:
                    model_input["image_input"] = [reference_image_url]
                    
            elif model_id == "nano-banana":
                model_input = {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",  # Square for avatars
                    "output_format": "png",
                }
                if reference_image_url:
                    model_input["image_input"] = [reference_image_url]
                    model_input["aspect_ratio"] = "match_input_image"  # Match reference
                    
            elif model_id == "p-image":
                model_input = {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",  # Square for avatars
                    "prompt_upsampling": False,  # Keep it simple/fast
                }
            else:
                model_input = {"prompt": prompt}
            
            # Start the prediction using the model-specific endpoint
            # Format: POST /v1/models/{owner}/{name}/predictions
            response = await client.post(
                f"https://api.replicate.com/v1/models/{model_name}/predictions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": model_input,
                },
            )
            
            if response.status_code not in (200, 201):
                error_text = response.text
                logger.error(f"Replicate API error: {error_text}")
                return {"success": False, "error": f"Replicate API error: {error_text}"}
            
            prediction = response.json()
            prediction_id = prediction.get("id")
            
            if not prediction_id:
                return {"success": False, "error": "No prediction ID returned"}
            
            logger.info(f"Started prediction: {prediction_id}")
            
            # Poll for completion
            for attempt in range(90):  # Max 90 attempts (3 minutes)
                await asyncio.sleep(2)
                
                status_response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                if status_response.status_code != 200:
                    continue
                
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "succeeded":
                    output = status_data.get("output")
                    
                    # Handle different output formats
                    if isinstance(output, list) and len(output) > 0:
                        image_url = output[0]
                    elif isinstance(output, str):
                        image_url = output
                    elif isinstance(output, dict) and "url" in output:
                        image_url = output["url"]
                    else:
                        logger.error(f"Unexpected output format: {output}")
                        return {"success": False, "error": f"Unexpected output format: {type(output)}"}
                    
                    logger.info(f"Generation succeeded: {image_url[:50]}...")
                    return {"success": True, "image_url": image_url}
                        
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    logger.error(f"Prediction failed: {error}")
                    return {"success": False, "error": error}
                    
                elif status == "canceled":
                    return {"success": False, "error": "Prediction was canceled"}
                
                # Log progress occasionally
                if attempt % 10 == 0:
                    logger.info(f"Still waiting... status: {status}")
            
            return {"success": False, "error": "Prediction timed out after 3 minutes"}
            
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        logger.error(f"Replicate generation error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Animation Generation
# =============================================================================

# Default animation settings for bytedance/seedance-1-pro-fast
DEFAULT_ANIMATION_SETTINGS = {
    "fps": 24,
    "prompt": "the person is talking",
    "duration": 5,
    "resolution": "480p",
    "aspect_ratio": "1:1",
    "camera_fixed": True,
}


async def generate_animation(
    image_url: str,
    prompt: str = "the person is talking",
    output_path: Optional[Path] = None,
    fps: int = 24,
    duration: int = 5,
) -> dict[str, Any]:
    """Generate an animated video from a static image using Replicate.

    Uses bytedance/seedance-1-pro-fast model to create a talking animation.

    Args:
        image_url: Public URL of the source image.
        prompt: Animation prompt describing the motion (e.g., "the person is talking").
        output_path: Optional path to save the video file.
        fps: Frames per second (default 24).
        duration: Duration in seconds (default 5).

    Returns:
        Dict with success status and video_url or error.
    """
    api_key = os.getenv("REPLICATE_API_TOKEN")
    if not api_key:
        return {"success": False, "error": "REPLICATE_API_TOKEN not set in environment"}

    model_name = "bytedance/seedance-1-pro-fast"
    
    logger.info(f"Generating animation: {prompt[:50]}...")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for video
            model_input = {
                "fps": fps,
                "prompt": prompt,
                "duration": duration,
                "resolution": "480p",
                "aspect_ratio": "1:1",
                "camera_fixed": True,
                "image": image_url,
            }
            
            # Start the prediction
            response = await client.post(
                f"https://api.replicate.com/v1/models/{model_name}/predictions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": model_input},
            )
            
            if response.status_code not in (200, 201):
                error_text = response.text
                logger.error(f"Replicate API error: {error_text}")
                return {"success": False, "error": f"Replicate API error: {error_text}"}
            
            prediction = response.json()
            prediction_id = prediction.get("id")
            
            if not prediction_id:
                return {"success": False, "error": "No prediction ID returned"}
            
            logger.info(f"Started animation prediction: {prediction_id}")
            
            # Poll for completion (video takes longer)
            for attempt in range(150):  # Max 5 minutes
                await asyncio.sleep(2)
                
                status_response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                
                if status_response.status_code != 200:
                    continue
                
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "succeeded":
                    output = status_data.get("output")
                    
                    # Get the video URL
                    if isinstance(output, str):
                        video_url = output
                    elif isinstance(output, dict) and "url" in output:
                        video_url = output["url"]
                    else:
                        logger.error(f"Unexpected output format: {output}")
                        return {"success": False, "error": f"Unexpected output format: {type(output)}"}
                    
                    logger.info(f"Animation generated: {video_url[:50]}...")
                    
                    # Download and save if output_path provided
                    if output_path:
                        video_response = await client.get(video_url)
                        video_response.raise_for_status()
                        output_path.write_bytes(video_response.content)
                        logger.info(f"Saved video to: {output_path}")
                    
                    return {
                        "success": True,
                        "video_url": video_url,
                        "saved_to": str(output_path) if output_path else None,
                    }
                    
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    logger.error(f"Animation prediction failed: {error}")
                    return {"success": False, "error": error}
                    
                elif status == "canceled":
                    return {"success": False, "error": "Prediction was canceled"}
                
                # Log progress occasionally
                if attempt % 15 == 0:
                    logger.info(f"Animation still generating... status: {status}")
            
            return {"success": False, "error": "Animation timed out after 5 minutes"}
            
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        logger.error(f"Animation generation error: {e}")
        return {"success": False, "error": str(e)}


def _run_ffmpeg_sync(
    palette_cmd: list[str],
    gif_cmd: list[str],
    simple_cmd: list[str],
    palette_path: Path,
    gif_path: Path,
) -> dict[str, Any]:
    """Synchronous ffmpeg execution (to be run in executor).

    This function handles the actual subprocess calls and palette cleanup.
    """
    try:
        # Run palette generation
        result = subprocess.run(
            palette_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error(f"Palette generation failed: {result.stderr}")
            # Try simpler one-pass conversion (no palette needed)
            result = subprocess.run(
                simple_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
        else:
            # Run GIF generation with palette
            try:
                result = subprocess.run(
                    gif_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    logger.error(f"GIF generation failed: {result.stderr}")
                    return {"success": False, "error": f"FFmpeg error: {result.stderr}"}
            finally:
                # Always clean up palette file after two-pass attempt
                if palette_path.exists():
                    palette_path.unlink()

        if gif_path.exists():
            logger.info(f"GIF created: {gif_path} ({gif_path.stat().st_size / 1024:.1f}KB)")
            return {
                "success": True,
                "gif_path": str(gif_path),
                "size_bytes": gif_path.stat().st_size,
            }
        else:
            return {"success": False, "error": "GIF file was not created"}

    except subprocess.TimeoutExpired:
        # Clean up palette on timeout too
        if palette_path.exists():
            palette_path.unlink()
        return {"success": False, "error": "FFmpeg conversion timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "FFmpeg not found. Please install ffmpeg."}
    except Exception as e:
        # Clean up palette on any error
        if palette_path.exists():
            palette_path.unlink()
        logger.error(f"GIF conversion error: {e}")
        return {"success": False, "error": str(e)}


async def convert_video_to_gif(
    video_path: Path,
    gif_path: Path,
    fps: int = 15,
    scale: int = 256,
) -> dict[str, Any]:
    """Convert a video file to an animated GIF using ffmpeg.

    Args:
        video_path: Path to the source video file.
        gif_path: Path to save the output GIF.
        fps: Output frame rate (default 15 for smaller file size).
        scale: Width in pixels (height auto-calculated, default 256).

    Returns:
        Dict with success status and gif_path or error.
    """
    if not video_path.exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    # Prepare paths and commands
    palette_path = video_path.parent / f"{video_path.stem}_palette.png"

    palette_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen",
        str(palette_path),
    ]

    gif_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(palette_path),
        "-lavfi", f"fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        str(gif_path),
    ]

    simple_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos",
        str(gif_path),
    ]

    # Run ffmpeg in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _run_ffmpeg_sync,
        palette_cmd,
        gif_cmd,
        simple_cmd,
        palette_path,
        gif_path,
    )


async def generate_expression_animation(
    image_path: Path,
    output_gif_path: Path,
    prompt: str = "the person is talking",
    public_base_url: Optional[str] = None,
    duration: int = 2,
) -> dict[str, Any]:
    """Generate an animated GIF from a static expression image.

    This is a high-level function that:
    1. Uploads/serves the image for Replicate to access
    2. Generates a video animation
    3. Converts the video to a looping GIF
    4. Cleans up temporary files

    Args:
        image_path: Path to the source expression PNG.
        output_gif_path: Path to save the output GIF.
        prompt: Animation prompt describing the motion.
        public_base_url: Public URL base for serving the image to Replicate.
        duration: Duration in seconds (default 2).

    Returns:
        Dict with success status and gif_path or error.
    """
    if not image_path.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}
    
    if not public_base_url:
        return {"success": False, "error": "Public URL required for Replicate to access the image"}
    
    # Build public URL for the image
    # Assuming image_path is like: agents/{agent_name}/expressions/{expression}.png
    relative_path = image_path.relative_to(Path("."))
    image_url = f"{public_base_url}/{relative_path}"
    
    logger.info(f"Generating animation from: {image_url}")
    
    # Create temp directory for video
    temp_video_path = image_path.parent / f"{image_path.stem}_temp.mp4"
    
    try:
        # Step 1: Generate video animation
        video_result = await generate_animation(
            image_url=image_url,
            prompt=prompt,
            output_path=temp_video_path,
            duration=duration,
        )
        
        if not video_result.get("success"):
            return video_result
        
        # Step 2: Convert video to GIF
        gif_result = await convert_video_to_gif(
            video_path=temp_video_path,
            gif_path=output_gif_path,
        )
        
        if not gif_result.get("success"):
            return gif_result
        
        return {
            "success": True,
            "gif_path": str(output_gif_path),
            "prompt": prompt,
        }
        
    finally:
        # Clean up temp video file
        if temp_video_path.exists():
            temp_video_path.unlink()
            logger.debug(f"Cleaned up temp video: {temp_video_path}")
