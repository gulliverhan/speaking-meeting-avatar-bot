# Recall.ai + ElevenLabs Meeting Bot

An AI meeting assistant that joins video calls via Recall.ai, powered by ElevenLabs Conversational AI. Features visual awareness (can see participants and screen shares), dynamic avatar expressions, and natural conversation.

## ✨ Features

### 🎤 Conversational AI
- **Natural speech** - Powered by ElevenLabs Conversational AI agents
- **Custom voices** - Configure any ElevenLabs voice per agent
- **Tool calling** - Agent can trigger actions via webhooks

### 👁️ Visual Awareness
- **See participants** - AI vision analyzes webcam feeds (clothing, background, expressions)
- **Screen share detection** - Automatically detects and describes shared content
- **Real-time updates** - `get_meeting_context` tool refreshes visual information
- **Smart caching** - Results cached for 30s to manage API costs

### 🎭 Avatar System
- **Dynamic expressions** - Avatar changes based on conversation (happy, thinking, interested, etc.)
- **Automatic speaking mode** - Expression switches when agent speaks
- **Custom avatars** - Generate avatars using AI image generation (Replicate)
- **Expression packs** - Generate consistent expression sets from a base image

### 🎮 Admin Dashboard
- **Deploy bots** - One-click deployment to any meeting URL
- **Agent management** - Create/edit agents with custom prompts, voices, and avatars
- **Live status** - Monitor active bots and their status
- **ElevenLabs sync** - Push local agent configs to ElevenLabs API

### 🔧 Agent Tools
- `set_expression` - Change avatar facial expression
- `get_meeting_context` - Get current visual context (participants + screen shares)

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Poetry](https://python-poetry.org/)
- [ngrok](https://ngrok.com/) (for local development)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd recall_elevenlabs

# Install dependencies
poetry install

# Copy environment template
cp env.example .env
```

### Configuration

Edit `.env` with your API keys:

```env
RECALL_API_KEY=your_recall_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
REPLICATE_API_TOKEN=your_replicate_token  # For avatar generation & vision
```

### Running Locally

```bash
# Start ngrok in a separate terminal
ngrok http 7014

# Run the server
poetry run python run.py --local-dev
```

Then open http://localhost:7014/static/admin.html

## 📁 Project Structure

```
recall_elevenlabs/
├── agents/                    # Agent configurations
│   ├── meeting_facilitator/   # Meeting facilitator agent
│   │   ├── config.yaml        # Agent config (tools, prompt, voice)
│   │   ├── expressions/       # Avatar expression images
│   │   ├── avatar_name.txt    # Display name
│   │   └── voice_id.txt       # ElevenLabs voice ID
│   └── case_study_interviewer/
├── app/                       # FastAPI application
│   ├── main.py               # App initialization
│   ├── routes.py             # API endpoints
│   ├── webhooks.py           # ElevenLabs tool webhooks
│   └── websockets.py         # Browser & video WebSocket handlers
├── core/                      # Core business logic
│   ├── bot_state.py          # Bot state management
│   ├── recall_client.py      # Recall.ai API client
│   ├── elevenlabs_client.py  # ElevenLabs API client
│   └── elevenlabs_agent_sync.py  # Sync agents to ElevenLabs
├── prompts/                   # Prompt templates
│   ├── participant_analysis.txt  # Vision analysis prompt
│   ├── default_system_prompt.txt # Default agent prompt
│   └── expression_modifiers.yaml # Expression descriptions
├── static/                    # Frontend files
│   ├── admin.html            # Admin dashboard
│   └── index.html            # Bot webpage (avatar display)
└── utils/                     # Utilities
    ├── image_generator.py    # Replicate image generation
    ├── prompts.py            # Prompt loading utilities
    └── ngrok.py              # ngrok tunnel detection
```

## 🎯 How It Works

1. **Bot Creation** - Admin deploys bot to a meeting URL via dashboard
2. **Recall.ai** - Bot joins meeting in a headless browser
3. **Video Streams** - Recall sends participant video frames via WebSocket
4. **AI Vision** - Frames analyzed by Gemini 2.5 Flash (via Replicate)
5. **Context Injection** - Visual descriptions sent to ElevenLabs agent
6. **Conversation** - Agent speaks using ElevenLabs, listens via browser
7. **Tool Calls** - Agent triggers webhooks for expressions, context refresh

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/bots` | GET | List active bots |
| `/bots` | POST | Create new bot |
| `/bots/{id}` | DELETE | Remove bot |
| `/agents` | GET | List available agents |
| `/agents/{name}/sync` | POST | Sync agent to ElevenLabs |
| `/webhooks/tools/set_expression` | POST | Set avatar expression |
| `/webhooks/tools/get_meeting_context` | GET | Get visual context |

## ⚙️ Configuration

### Agent Config (`agents/{name}/config.yaml`)

```yaml
name: "Meeting Facilitator"
voice_id: "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs voice ID
first_message: "Hey! What's up?"

tools:
  - set_expression
  - get_meeting_context

expressions:
  - neutral
  - happy
  - thinking
  - interested
  - speaking

system_prompt: |
  You are a professional meeting facilitator...
```

### Custom Avatar

1. Go to Admin Dashboard → Agents tab
2. Select an agent
3. Write a prompt and generate base avatar
4. Generate expression pack from base
5. (Optional) Upload a reference image for consistency

## 🔄 Frame & Context Caching

To manage API costs:
- **Video frames** saved every 30 seconds per stream
- **Context results** cached for 30 seconds
- Repeated `get_meeting_context` calls return cached results

## 🛠️ Development

```bash
# Run with auto-reload
poetry run python run.py --local-dev

# The server will:
# - Auto-detect ngrok tunnels
# - Enable hot-reload on file changes
# - Serve admin UI at /static/admin.html
```

## 📝 License

MIT
