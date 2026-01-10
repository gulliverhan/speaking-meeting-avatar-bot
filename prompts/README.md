# Prompts

This folder contains all the prompt templates used throughout the application. This makes it easy to edit and experiment with prompts without modifying code.

## Files

### Avatar Generation
- `avatar_from_reference.txt` - Prompt for generating avatars when a reference image is provided
- `avatar_from_scratch.txt` - Prompt for generating avatars from a text description only

### Expression Generation
- `expression_modifiers.yaml` - YAML file containing expression modifiers (happy, thinking, etc.)
- `expression_from_reference.txt` - Prompt for generating expressions when using a base avatar as reference
- `expression_from_scratch.txt` - Prompt for generating expressions from text description only
- `single_expression_from_reference.txt` - Prompt for generating a single expression with reference
- `single_expression_from_scratch.txt` - Prompt for generating a single expression from scratch

### AI Analysis
- `participant_analysis.txt` - Prompt for analyzing participant video frames

### System
- `default_system_prompt.txt` - Default fallback system prompt for agents

## Template Variables

Prompts use Python's `str.format()` syntax for variable substitution:

```
{variable_name}
```

Common variables:
- `{prompt}` - User-provided description
- `{modifier}` - Expression modifier (e.g., "with a happy smile")
- `{person_desc}` - Description of the person
- `{base_prompt}` - Base character description
- `{expression}` - The expression name
- `{participant_name}` - Name of a meeting participant

## Usage in Code

```python
from utils.prompts import format_prompt, get_expression_modifier, load_prompt

# Load and format a prompt
prompt = format_prompt("avatar_from_reference", prompt="A friendly businesswoman")

# Get an expression modifier
modifier = get_expression_modifier("happy")  # Returns "with a warm, genuine smile..."

# Load a raw prompt without formatting
raw = load_prompt("default_system_prompt")
```

## Adding New Prompts

1. Create a `.txt` or `.md` file in this folder
2. Use `{variable_name}` for any dynamic content
3. Import and use via `format_prompt("your_file_name", var1=val1, var2=val2)`

For YAML-based prompts (like expression modifiers), use `load_yaml_prompts("filename")`.
