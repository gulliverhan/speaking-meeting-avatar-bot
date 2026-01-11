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
- **Animated avatars** - Each expression has idle (listening) and speaking GIF animations
- **Custom avatars** - Generate avatars using AI image generation (Replicate)
- **Expression packs** - Generate consistent expression sets from a base image
- **Smart prompts** - Idle animation prompts are saved and auto-inherited by speaking animations

### 🎮 Admin Dashboard
- **Deploy bots** - One-click deployment to any meeting URL
- **Agent management** - Create/edit agents with custom prompts, voices, and avatars
- **Live status** - Monitor active bots and their status
- **ElevenLabs sync** - Push local agent configs to ElevenLabs API
- **Statistics tab** - View conversation metrics and compare provider performance

### 📊 Instrumentation & Metrics
- **Turn-around time (TAT)** - Measures response latency
- **Overlap detection** - Identifies when user and bot speak simultaneously  
- **Interruption tracking** - Counts when user cuts off the bot
- **Provider comparison** - Compare performance across different voice AI backends
- **SQLite persistence** - Metrics survive server restarts
- **Real-time dashboard** - Visualize metrics in the admin UI

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
│   ├── meeting_facilitator/   # Example agent
│   │   ├── config.yaml        # Agent config (tools, prompt, voice)
│   │   ├── expressions/       # Avatar expression images & animations
│   │   │   ├── neutral.png    # Static expression (required)
│   │   │   ├── neutral.gif    # Idle animation (optional)
│   │   │   ├── neutral_speaking.gif  # Speaking animation (optional)
│   │   │   └── ...
│   │   ├── custom_animation_prompts.yaml  # Saved animation prompts
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
│   ├── elevenlabs_agent_sync.py  # Sync agents to ElevenLabs
│   └── instrumentation.py    # Metrics collection & persistence
├── data/                      # Persistent data
│   └── metrics.db            # SQLite database for metrics
├── prompts/                   # Prompt templates
│   ├── animation_prompts.yaml    # Default animation prompts per expression
│   ├── participant_analysis.txt  # Vision analysis prompt
│   ├── default_system_prompt.txt # Default agent prompt
│   └── expression_modifiers.yaml # Expression descriptions
├── static/                    # Frontend files
│   ├── admin.html            # Admin dashboard
│   └── index.html            # Bot webpage (avatar display)
└── utils/                     # Utilities
    ├── image_generator.py    # Replicate image/animation generation
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
| `/bots/{id}/metrics` | GET | Get metrics for a specific bot |
| `/agents` | GET | List available agents |
| `/agents/{name}/sync` | POST | Sync agent to ElevenLabs |
| `/agents/avatar-status` | GET | Get avatar/animation status for all agents |
| `/agents/{name}/generate-animation` | POST | Generate idle or speaking animation |
| `/agents/{name}/expressions/{expr}/animation-prompt` | GET | Get animation prompt (with saved customizations) |
| `/metrics` | GET | Get all conversation metrics |
| `/metrics/export` | GET | Export metrics (JSON/CSV) |
| `/conversations/{id}/events` | GET | Get events for a conversation |
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
  - curious

system_prompt: |
  You are a professional meeting facilitator...
```

### Custom Avatar

1. Go to Admin Dashboard → Agents tab
2. Select an agent
3. Write a prompt and generate base avatar
4. Generate expression pack from base
5. (Optional) Upload a reference image for consistency

### Animated Expressions

Each expression can have three visual states:
- **Static PNG** (required) - Fallback when no animation exists
- **Idle GIF** (optional) - Played when listening/waiting
- **Speaking GIF** (optional) - Played when the agent is talking

**Generating animations:**

1. In the Admin Dashboard, each expression card shows two buttons:
   - 👁️ Generate idle animation (when listening)
   - 🗣️ Generate speaking animation (when talking)

2. When generating an **idle** animation:
   - Customize the prompt (e.g., "professional woman listening calmly, subtle breathing")
   - Your prompt is saved for this expression

3. When generating a **speaking** animation:
   - The saved idle prompt is auto-loaded with speaking modifiers added
   - This ensures visual consistency between idle and speaking states

**Animation prompts are stored in:**
- `agents/{name}/custom_animation_prompts.yaml` - Per-agent saved prompts
- `prompts/animation_prompts.yaml` - Default prompts per expression type

**Example `animation_prompts.yaml`:**
```yaml
neutral:
  idle: "the person is listening calmly with subtle breathing and eye movements"
  speaking: "the person is talking calmly with natural mouth movements"
happy:
  idle: "the person is listening with a warm smile, nodding gently"
  speaking: "the person is talking happily with animated mouth movements"
```

## 📊 Instrumentation System

The instrumentation layer measures conversation quality metrics to help compare voice AI backends (ElevenLabs, Pipecat, etc.) and identify performance issues.

### Metrics Collected

| Metric | Description | How It's Measured |
|--------|-------------|-------------------|
| **Turn-Around Time (TAT)** | Time from user stops speaking to bot starts speaking | `bot_speech_start_ms - user_speech_end_ms` |
| **Average TAT** | Mean response latency across all turns | Average of all TAT values in conversation |
| **P50 TAT** | Median response latency | 50th percentile of TAT values |
| **P95 TAT** | Worst-case latency (95th percentile) | 95th percentile of TAT values |
| **Overlaps** | Instances of simultaneous speech | Detected when both user and bot are marked as "speaking" |
| **Interruptions** | User cuts off bot mid-sentence | Bot speech ended early due to user input |
| **Total Turns** | Number of back-and-forth exchanges | Count of complete user→bot turn cycles |

### Event Types Tracked

The system records these events with millisecond timestamps:

```
conversation_start     → Session begins
user_speech_start      → User begins speaking (from transcription)
user_speech_end        → User stops speaking (final transcript)
transcription_received → Speech-to-text result arrives
bot_speech_start       → Bot audio begins playing
bot_speech_end         → Bot finishes speaking
agent_response_text    → LLM response text received
interruption           → User interrupted the bot
overlap_start          → Both parties speaking simultaneously
conversation_end       → Session ends
```

### How Metrics Are Calculated

**Turn-Around Time (TAT):**
```
TAT = bot_speech_start_timestamp - user_speech_end_timestamp
```
- Positive TAT = Normal response delay
- Negative TAT = Bot started before user finished (overlap/interruption)
- Good: < 500ms | Warning: 500-1500ms | Bad: > 1500ms

**Overlap Detection:**
- Triggered when `user_speech_start` arrives while bot is still speaking
- Only counted when we have actual evidence of user speech (transcription received)
- Silence during bot speech does NOT count as overlap

**Quality Benchmarks:**

| Rating | TAT | Description |
|--------|-----|-------------|
| 🟢 Excellent | < 300ms | Near-instantaneous response |
| 🟢 Good | 300-500ms | Natural conversational pace |
| 🟡 Acceptable | 500-1000ms | Noticeable but tolerable delay |
| 🟡 Slow | 1-2s | Awkward pauses |
| 🔴 Poor | > 2s | Conversation feels broken |

### Data Storage

Metrics are persisted to SQLite at `data/metrics.db`:

```sql
-- Raw events (source of truth)
events (timestamp_ms, event_type, conversation_id, bot_id, metadata)

-- Conversation summaries
conversations (conversation_id, bot_id, backend_type, total_turns, 
               avg_tat_ms, p50_tat_ms, p95_tat_ms, overlap_count, ...)

-- Per-turn metrics
turns (conversation_id, turn_id, user_speech_start_ms, bot_speech_start_ms,
       turn_around_time_ms, had_overlap, was_interrupted, ...)
```

### Viewing Metrics

**Admin Dashboard:**
1. Go to http://localhost:7014/static/admin.html
2. Click the **📊 Statistics** tab
3. View summary stats, latency distribution, and conversation history
4. Filter by provider to compare backends

**API Endpoints:**
```bash
# Get all conversation metrics
GET /metrics

# Export metrics as JSON or CSV
GET /metrics/export?format=json

# Get metrics for specific bot
GET /bots/{bot_id}/metrics

# Get events for a conversation
GET /conversations/{conv_id}/events
```

### Provider Comparison

When using multiple backends (ElevenLabs, Pipecat, etc.), the dashboard shows:
- Side-by-side comparison cards
- "⚡ Fastest" badge on best performer
- "🐢 Slowest" badge on worst performer
- Filter dropdown to view metrics by provider

### Architecture

```
Browser SDK                    Server
    │                            │
    ├─ onModeChange ────────────►│
    │  (speaking/listening)      │
    │                            │
    ├─ onMessage ───────────────►│ ─── MetricsCollector
    │  (transcripts)             │         │
    │                            │         ▼
    └─ sendMetricEvent ─────────►│ ─── SQLite (data/metrics.db)
       (via WebSocket)           │
```

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
