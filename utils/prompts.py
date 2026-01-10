"""Utility for loading prompts from the prompts/ folder."""

from pathlib import Path
from functools import lru_cache
from typing import Optional

import yaml

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts folder.
    
    Args:
        name: The prompt name (without extension). Will try .txt first, then .md.
        
    Returns:
        The prompt template string, or empty string if not found.
    """
    # Try .txt extension first
    txt_path = PROMPTS_DIR / f"{name}.txt"
    if txt_path.exists():
        return txt_path.read_text().strip()
    
    # Try .md extension
    md_path = PROMPTS_DIR / f"{name}.md"
    if md_path.exists():
        return md_path.read_text().strip()
    
    return ""


@lru_cache(maxsize=16)
def load_yaml_prompts(name: str) -> dict:
    """Load prompts from a YAML file.
    
    Args:
        name: The YAML file name (without extension).
        
    Returns:
        Dictionary of prompts, or empty dict if not found.
    """
    yaml_path = PROMPTS_DIR / f"{name}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_expression_modifier(expression: str) -> str:
    """Get the modifier prompt for a specific expression.
    
    Args:
        expression: The expression name (e.g., 'happy', 'thinking').
        
    Returns:
        The modifier string for that expression.
    """
    modifiers = load_yaml_prompts("expression_modifiers")
    return modifiers.get(expression, modifiers.get("neutral", "with a neutral expression"))


def format_prompt(template_name: str, **kwargs) -> str:
    """Load and format a prompt template with the given variables.
    
    Args:
        template_name: The prompt template name.
        **kwargs: Variables to substitute into the template.
        
    Returns:
        The formatted prompt string.
    """
    template = load_prompt(template_name)
    if not template:
        return ""
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Return template as-is if formatting fails
        return template


def clear_cache():
    """Clear all cached prompts (useful after editing prompt files)."""
    load_prompt.cache_clear()
    load_yaml_prompts.cache_clear()
