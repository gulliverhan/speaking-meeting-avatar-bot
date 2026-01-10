"""Agent configuration management.

This module handles loading and managing agent configurations from the
agents/ directory.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from utils.logger import logger

# Cache for loaded agent configs
_agent_configs: dict[str, dict[str, Any]] = {}


def get_agents_dir() -> Path:
    """Get the path to the agents directory.

    Returns:
        Path to the agents directory.
    """
    # Look for agents dir relative to this file's location
    current_dir = Path(__file__).parent.parent
    agents_dir = current_dir / "agents"

    if not agents_dir.exists():
        # Try from working directory
        agents_dir = Path.cwd() / "agents"

    return agents_dir


def list_agents() -> list[str]:
    """List all available agent configurations.

    Returns:
        List of agent names (directory names).
    """
    agents_dir = get_agents_dir()

    if not agents_dir.exists():
        logger.warning(f"Agents directory not found: {agents_dir}")
        return []

    return [
        d.name
        for d in agents_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def load_agent_config(agent_name: str) -> Optional[dict[str, Any]]:
    """Load an agent's configuration from its config.yaml file.

    Args:
        agent_name: Name of the agent (directory name).

    Returns:
        Agent configuration dictionary, or None if not found.
    """
    agents_dir = get_agents_dir()
    config_path = agents_dir / agent_name / "config.yaml"

    if not config_path.exists():
        logger.warning(f"Agent config not found: {config_path}")
        return None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded agent config: {agent_name}")
        return config

    except Exception as e:
        logger.error(f"Error loading agent config {agent_name}: {e}")
        return None


def get_agent_config(agent_name: str) -> Optional[dict[str, Any]]:
    """Get an agent's configuration, using cache if available.

    Args:
        agent_name: Name of the agent.

    Returns:
        Agent configuration dictionary, or None if not found.
    """
    if agent_name not in _agent_configs:
        config = load_agent_config(agent_name)
        if config:
            _agent_configs[agent_name] = config

    return _agent_configs.get(agent_name)


def get_agent_expressions_dir(agent_name: str) -> Optional[Path]:
    """Get the path to an agent's expressions directory.

    Args:
        agent_name: Name of the agent.

    Returns:
        Path to expressions directory, or None if not found.
    """
    agents_dir = get_agents_dir()
    expressions_dir = agents_dir / agent_name / "expressions"

    if expressions_dir.exists():
        return expressions_dir
    return None


def get_agent_overlays_dir(agent_name: str) -> Optional[Path]:
    """Get the path to an agent's overlays directory.

    Args:
        agent_name: Name of the agent.

    Returns:
        Path to overlays directory, or None if not found.
    """
    agents_dir = get_agents_dir()
    overlays_dir = agents_dir / agent_name / "overlays"

    if overlays_dir.exists():
        return overlays_dir
    return None


def clear_agent_cache() -> None:
    """Clear the agent configuration cache."""
    _agent_configs.clear()
    logger.info("Agent config cache cleared")
