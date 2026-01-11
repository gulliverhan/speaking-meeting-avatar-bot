"""Voice AI instrumentation for measuring latency and conversation quality.

This module provides comprehensive metrics collection for voice AI backends,
enabling comparison between ElevenLabs, Pipecat, and other providers.

Key Metrics:
- Turn-Around Time (TAT): User stops speaking → Bot starts speaking
- Time to First Byte (TTFB): Audio sent → First response audio
- Overlap Detection: Both parties speaking simultaneously
- Interruption Tracking: User interrupts bot mid-speech

Persistence:
- Events and metrics are persisted to SQLite for historical analysis
- Data survives server restarts
- Database file: data/metrics.db
"""

import asyncio
import json
import os
import sqlite3
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from threading import Lock

from utils.logger import logger


# Database configuration
DEFAULT_DB_PATH = Path("data/metrics.db")


class MetricsDatabase:
    """SQLite database for persisting instrumentation events and metrics.
    
    Thread-safe database access with automatic schema creation.
    """
    
    SCHEMA = """
    -- Raw events table - source of truth
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_ms REAL NOT NULL,
        event_type TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        bot_id TEXT NOT NULL,
        turn_id INTEGER,
        metadata TEXT,  -- JSON
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_events_conversation ON events(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_events_bot ON events(bot_id);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_ms);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    
    -- Conversation summary table for fast lookups
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        bot_id TEXT NOT NULL,
        backend_type TEXT DEFAULT 'elevenlabs',
        start_time_ms REAL,
        end_time_ms REAL,
        total_turns INTEGER DEFAULT 0,
        avg_tat_ms REAL,
        p50_tat_ms REAL,
        p95_tat_ms REAL,
        overlap_count INTEGER DEFAULT 0,
        total_overlap_ms REAL DEFAULT 0,
        interruption_count INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_conversations_bot ON conversations(bot_id);
    
    -- Turn metrics table
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        turn_id INTEGER NOT NULL,
        user_speech_start_ms REAL,
        user_speech_end_ms REAL,
        first_audio_byte_ms REAL,
        bot_speech_start_ms REAL,
        bot_speech_end_ms REAL,
        transcription_received_ms REAL,
        audio_sent_ms REAL,
        turn_around_time_ms REAL,
        time_to_first_byte_ms REAL,
        transcription_latency_ms REAL,
        full_response_time_ms REAL,
        had_overlap INTEGER DEFAULT 0,
        overlap_duration_ms REAL DEFAULT 0,
        was_interrupted INTEGER DEFAULT 0,
        UNIQUE(conversation_id, turn_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id);
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._lock = Lock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database and create tables."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()
        
        logger.info(f"Metrics database initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_event(self, event: "ConversationEvent") -> int:
        """Save an event to the database.
        
        Args:
            event: The event to save.
            
        Returns:
            The database row ID of the saved event.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO events (timestamp_ms, event_type, conversation_id, bot_id, turn_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp_ms,
                        event.event_type,
                        event.conversation_id,
                        event.bot_id,
                        event.turn_id,
                        json.dumps(event.metadata) if event.metadata else None,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
    
    def save_conversation(self, metrics: "ConversationMetrics") -> None:
        """Save or update conversation summary metrics.
        
        Args:
            metrics: The conversation metrics to save.
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, bot_id, backend_type, start_time_ms, end_time_ms,
                     total_turns, avg_tat_ms, p50_tat_ms, p95_tat_ms,
                     overlap_count, total_overlap_ms, interruption_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        metrics.conversation_id,
                        metrics.bot_id,
                        metrics.backend_type,
                        metrics.start_time_ms,
                        metrics.end_time_ms,
                        metrics.total_turns,
                        metrics.avg_tat_ms,
                        metrics.p50_tat_ms,
                        metrics.p95_tat_ms,
                        metrics.overlap_count,
                        metrics.total_overlap_ms,
                        metrics.interruption_count,
                    ),
                )
                conn.commit()
    
    def save_turn(self, conversation_id: str, turn: "TurnMetrics") -> None:
        """Save or update a turn's metrics.
        
        Args:
            conversation_id: The conversation ID.
            turn: The turn metrics to save.
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO turns
                    (conversation_id, turn_id, user_speech_start_ms, user_speech_end_ms,
                     first_audio_byte_ms, bot_speech_start_ms, bot_speech_end_ms,
                     transcription_received_ms, audio_sent_ms, turn_around_time_ms,
                     time_to_first_byte_ms, transcription_latency_ms, full_response_time_ms,
                     had_overlap, overlap_duration_ms, was_interrupted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        turn.turn_id,
                        turn.user_speech_start_ms,
                        turn.user_speech_end_ms,
                        turn.first_audio_byte_ms,
                        turn.bot_speech_start_ms,
                        turn.bot_speech_end_ms,
                        turn.transcription_received_ms,
                        turn.audio_sent_ms,
                        turn.turn_around_time_ms,
                        turn.time_to_first_byte_ms,
                        turn.transcription_latency_ms,
                        turn.full_response_time_ms,
                        1 if turn.had_overlap else 0,
                        turn.overlap_duration_ms,
                        1 if turn.was_interrupted else 0,
                    ),
                )
                conn.commit()
    
    def load_events(self, conversation_id: str) -> list["ConversationEvent"]:
        """Load all events for a conversation.
        
        Args:
            conversation_id: The conversation ID.
            
        Returns:
            List of events in chronological order.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_ms, event_type, conversation_id, bot_id, turn_id, metadata
                FROM events
                WHERE conversation_id = ?
                ORDER BY timestamp_ms
                """,
                (conversation_id,),
            ).fetchall()
        
        events = []
        for row in rows:
            events.append(ConversationEvent(
                timestamp_ms=row["timestamp_ms"],
                event_type=row["event_type"],
                conversation_id=row["conversation_id"],
                bot_id=row["bot_id"],
                turn_id=row["turn_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ))
        return events
    
    def load_conversation(self, conversation_id: str) -> Optional["ConversationMetrics"]:
        """Load conversation metrics from database.
        
        Args:
            conversation_id: The conversation ID.
            
        Returns:
            ConversationMetrics or None if not found.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversations WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            
            if not row:
                return None
            
            # Load turns for this conversation
            turn_rows = conn.execute(
                """
                SELECT * FROM turns WHERE conversation_id = ? ORDER BY turn_id
                """,
                (conversation_id,),
            ).fetchall()
        
        turns = []
        for tr in turn_rows:
            turns.append(TurnMetrics(
                turn_id=tr["turn_id"],
                user_speech_start_ms=tr["user_speech_start_ms"],
                user_speech_end_ms=tr["user_speech_end_ms"],
                first_audio_byte_ms=tr["first_audio_byte_ms"],
                bot_speech_start_ms=tr["bot_speech_start_ms"],
                bot_speech_end_ms=tr["bot_speech_end_ms"],
                transcription_received_ms=tr["transcription_received_ms"],
                audio_sent_ms=tr["audio_sent_ms"],
                turn_around_time_ms=tr["turn_around_time_ms"],
                time_to_first_byte_ms=tr["time_to_first_byte_ms"],
                transcription_latency_ms=tr["transcription_latency_ms"],
                full_response_time_ms=tr["full_response_time_ms"],
                had_overlap=bool(tr["had_overlap"]),
                overlap_duration_ms=tr["overlap_duration_ms"] or 0.0,
                was_interrupted=bool(tr["was_interrupted"]),
            ))
        
        metrics = ConversationMetrics(
            conversation_id=row["conversation_id"],
            bot_id=row["bot_id"],
            backend_type=row["backend_type"],
            start_time_ms=row["start_time_ms"],
            end_time_ms=row["end_time_ms"],
            turns=turns,
        )
        return metrics
    
    def load_all_conversation_ids(self) -> list[str]:
        """Load all conversation IDs from database.
        
        Returns:
            List of conversation IDs.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT conversation_id FROM conversations ORDER BY start_time_ms DESC"
            ).fetchall()
        return [row["conversation_id"] for row in rows]
    
    def load_bot_conversations(self, bot_id: str) -> list[str]:
        """Load conversation IDs for a specific bot.
        
        Args:
            bot_id: The bot client ID.
            
        Returns:
            List of conversation IDs, most recent first.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id FROM conversations 
                WHERE bot_id = ? 
                ORDER BY start_time_ms DESC
                """,
                (bot_id,),
            ).fetchall()
        return [row["conversation_id"] for row in rows]
    
    def get_stats(self) -> dict:
        """Get database statistics.
        
        Returns:
            Dictionary with event count, conversation count, etc.
        """
        with self._get_connection() as conn:
            event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            turn_count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        
        return {
            "event_count": event_count,
            "conversation_count": conv_count,
            "turn_count": turn_count,
            "db_path": str(self.db_path),
        }


class EventType(str, Enum):
    """Types of instrumentation events."""
    
    # User/Meeting audio events
    USER_AUDIO_RECEIVED = "user_audio_received"
    USER_SPEECH_START = "user_speech_start"
    USER_SPEECH_END = "user_speech_end"
    
    # Backend communication events
    AUDIO_SENT_TO_BACKEND = "audio_sent_to_backend"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    AGENT_RESPONSE_START = "agent_response_start"
    AGENT_RESPONSE_TEXT = "agent_response_text"
    
    # Bot audio events
    FIRST_AUDIO_BYTE = "first_audio_byte"
    BOT_SPEECH_START = "bot_speech_start"
    BOT_SPEECH_END = "bot_speech_end"
    AUDIO_SENT_TO_MEETING = "audio_sent_to_meeting"
    
    # Conversation flow events
    INTERRUPTION = "interruption"
    OVERLAP_START = "overlap_start"
    OVERLAP_END = "overlap_end"
    
    # Session events
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    
    # Pipecat-specific events (for future use)
    STT_START = "stt_start"
    STT_END = "stt_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    TTS_START = "tts_start"
    TTS_END = "tts_end"


@dataclass
class ConversationEvent:
    """A single instrumentation event in a conversation.
    
    Attributes:
        timestamp_ms: Unix timestamp in milliseconds.
        event_type: Type of event (from EventType enum).
        conversation_id: Unique conversation identifier.
        bot_id: Bot client ID.
        turn_id: Optional turn number within conversation.
        metadata: Backend-specific additional data.
    """
    timestamp_ms: float
    event_type: str
    conversation_id: str
    bot_id: str
    turn_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        return json.dumps(self.to_dict())


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn (user speaks, bot responds).
    
    Attributes:
        turn_id: Turn number within conversation.
        user_speech_start_ms: When user started speaking.
        user_speech_end_ms: When user stopped speaking.
        first_audio_byte_ms: When first bot audio was received.
        bot_speech_start_ms: When bot audio started playing.
        bot_speech_end_ms: When bot finished speaking.
        transcription_received_ms: When transcription event arrived.
        turn_around_time_ms: TAT (user end → bot start).
        time_to_first_byte_ms: TTFB (audio sent → first audio received).
        had_overlap: Whether overlap occurred during this turn.
        overlap_duration_ms: Total overlap time if any.
        was_interrupted: Whether user interrupted the bot.
    """
    turn_id: int
    user_speech_start_ms: Optional[float] = None
    user_speech_end_ms: Optional[float] = None
    first_audio_byte_ms: Optional[float] = None
    bot_speech_start_ms: Optional[float] = None
    bot_speech_end_ms: Optional[float] = None
    transcription_received_ms: Optional[float] = None
    audio_sent_ms: Optional[float] = None
    
    # Calculated metrics
    turn_around_time_ms: Optional[float] = None
    time_to_first_byte_ms: Optional[float] = None
    transcription_latency_ms: Optional[float] = None
    full_response_time_ms: Optional[float] = None
    
    # Quality metrics
    had_overlap: bool = False
    overlap_duration_ms: float = 0.0
    was_interrupted: bool = False
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics from raw timestamps."""
        # Turn-around time: user stops → bot starts
        if self.user_speech_end_ms and self.bot_speech_start_ms:
            self.turn_around_time_ms = self.bot_speech_start_ms - self.user_speech_end_ms
        
        # Time to first byte: audio sent → first audio received
        if self.audio_sent_ms and self.first_audio_byte_ms:
            self.time_to_first_byte_ms = self.first_audio_byte_ms - self.audio_sent_ms
        
        # Transcription latency: audio sent → transcription received
        if self.audio_sent_ms and self.transcription_received_ms:
            self.transcription_latency_ms = self.transcription_received_ms - self.audio_sent_ms
        
        # Full response time: user stops → bot stops
        if self.user_speech_end_ms and self.bot_speech_end_ms:
            self.full_response_time_ms = self.bot_speech_end_ms - self.user_speech_end_ms
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ConversationMetrics:
    """Aggregated metrics for an entire conversation.
    
    Attributes:
        conversation_id: Unique conversation identifier.
        bot_id: Bot client ID.
        backend_type: Voice AI backend ("elevenlabs" or "pipecat").
        start_time_ms: When conversation started.
        end_time_ms: When conversation ended (if ended).
        turns: List of per-turn metrics.
    """
    conversation_id: str
    bot_id: str
    backend_type: str = "elevenlabs"
    start_time_ms: Optional[float] = None
    end_time_ms: Optional[float] = None
    turns: list[TurnMetrics] = field(default_factory=list)
    
    @property
    def total_turns(self) -> int:
        """Total number of conversation turns."""
        return len(self.turns)
    
    @property
    def turn_around_times(self) -> list[float]:
        """List of all TAT values (excluding None)."""
        return [t.turn_around_time_ms for t in self.turns if t.turn_around_time_ms is not None]
    
    @property
    def avg_tat_ms(self) -> Optional[float]:
        """Average turn-around time in milliseconds."""
        tats = self.turn_around_times
        return statistics.mean(tats) if tats else None
    
    @property
    def p50_tat_ms(self) -> Optional[float]:
        """Median (p50) turn-around time."""
        tats = self.turn_around_times
        return statistics.median(tats) if tats else None
    
    @property
    def p95_tat_ms(self) -> Optional[float]:
        """95th percentile turn-around time."""
        tats = sorted(self.turn_around_times)
        if not tats:
            return None
        idx = int(len(tats) * 0.95)
        return tats[min(idx, len(tats) - 1)]
    
    @property
    def overlap_count(self) -> int:
        """Number of turns with overlap."""
        return sum(1 for t in self.turns if t.had_overlap)
    
    @property
    def total_overlap_ms(self) -> float:
        """Total overlap duration across all turns."""
        return sum(t.overlap_duration_ms for t in self.turns)
    
    @property
    def interruption_count(self) -> int:
        """Number of interruptions."""
        return sum(1 for t in self.turns if t.was_interrupted)
    
    def to_summary_dict(self) -> dict:
        """Convert to summary dictionary for API response."""
        return {
            "conversation_id": self.conversation_id,
            "bot_id": self.bot_id,
            "backend_type": self.backend_type,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "total_turns": self.total_turns,
            "avg_tat_ms": self.avg_tat_ms,
            "p50_tat_ms": self.p50_tat_ms,
            "p95_tat_ms": self.p95_tat_ms,
            "overlap_count": self.overlap_count,
            "total_overlap_ms": self.total_overlap_ms,
            "interruption_count": self.interruption_count,
            "turn_around_times": self.turn_around_times,
        }
    
    def to_full_dict(self) -> dict:
        """Convert to full dictionary including all turns."""
        result = self.to_summary_dict()
        result["turns"] = [t.to_dict() for t in self.turns]
        return result


class ConversationStore:
    """Thread-safe storage for conversation events and metrics.
    
    Stores events and metrics per conversation, with automatic
    calculation of turn metrics when events are recorded.
    
    Data is persisted to SQLite for historical analysis.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self._lock = Lock()
        self._events: dict[str, list[ConversationEvent]] = defaultdict(list)
        self._metrics: dict[str, ConversationMetrics] = {}
        self._current_turn: dict[str, TurnMetrics] = {}
        self._speaking_state: dict[str, dict] = {}  # bot_id -> {user: bool, bot: bool, overlap_active: bool, overlap_start_ts: float}
        self._subscribers: list[Callable[[ConversationEvent], None]] = []
        self._pending_overlap_events: list[ConversationEvent] = []  # Overlap events to persist outside lock
        
        # Initialize database
        self._db = MetricsDatabase(db_path)
        
        # Load existing conversation IDs from database
        self._load_conversation_index()
    
    def _load_conversation_index(self) -> None:
        """Load conversation IDs from database on startup.
        
        This populates the in-memory index so we know what's available.
        Full data is loaded lazily when requested.
        """
        try:
            conv_ids = self._db.load_all_conversation_ids()
            # Just mark these as known (data loaded on demand)
            for conv_id in conv_ids:
                if conv_id not in self._metrics:
                    # Placeholder - will be fully loaded when accessed
                    self._metrics[conv_id] = None
            
            stats = self._db.get_stats()
            logger.info(
                f"Loaded metrics index: {stats['conversation_count']} conversations, "
                f"{stats['event_count']} events from {stats['db_path']}"
            )
        except Exception as e:
            logger.error(f"Error loading conversation index: {e}")
    
    def subscribe(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Subscribe to receive all events in real-time.
        
        Args:
            callback: Function called with each new event.
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Unsubscribe from event notifications."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, event: ConversationEvent) -> None:
        """Notify all subscribers of a new event."""
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
    
    def record_event(self, event: ConversationEvent) -> None:
        """Record an instrumentation event.
        
        Args:
            event: The event to record.
        """
        with self._lock:
            conv_id = event.conversation_id
            bot_id = event.bot_id
            
            # Store the raw event in memory
            self._events[conv_id].append(event)
            
            # Initialize metrics if needed (or load from DB if placeholder)
            if conv_id not in self._metrics or self._metrics[conv_id] is None:
                self._metrics[conv_id] = ConversationMetrics(
                    conversation_id=conv_id,
                    bot_id=bot_id,
                    backend_type=event.metadata.get("backend_type", "elevenlabs"),
                )
            
            # Initialize speaking state if needed
            if bot_id not in self._speaking_state:
                self._speaking_state[bot_id] = {
                    "user": False, 
                    "bot": False,
                    "overlap_active": False,
                    "overlap_start_ts": None
                }
            
            # Process the event
            self._process_event(event)
            
            # Grab any pending overlap events to persist outside lock
            pending_overlap = list(self._pending_overlap_events)
            self._pending_overlap_events.clear()
        
        # Persist to database (outside lock for better concurrency)
        try:
            self._db.save_event(event)
            # Also persist any overlap events that were generated
            for overlap_event in pending_overlap:
                self._db.save_event(overlap_event)
        except Exception as e:
            logger.error(f"Error saving event to database: {e}")
        
        # Notify subscribers (outside lock to avoid deadlocks)
        self._notify_subscribers(event)
        for overlap_event in pending_overlap:
            self._notify_subscribers(overlap_event)
        
        # Log the event as structured JSON
        self._log_event(event)
        for overlap_event in pending_overlap:
            self._log_event(overlap_event)
    
    def _process_event(self, event: ConversationEvent) -> None:
        """Process an event and update metrics.
        
        Called with lock held.
        """
        conv_id = event.conversation_id
        bot_id = event.bot_id
        metrics = self._metrics[conv_id]
        speaking = self._speaking_state[bot_id]
        
        event_type = event.event_type
        ts = event.timestamp_ms
        
        # Handle conversation lifecycle
        if event_type == EventType.CONVERSATION_START:
            metrics.start_time_ms = ts
        elif event_type == EventType.CONVERSATION_END:
            metrics.end_time_ms = ts
            # Finalize any in-progress turn
            self._finalize_turn(conv_id)
        
        # Handle user speech events
        elif event_type == EventType.USER_SPEECH_START:
            speaking["user"] = True
            # Start a new turn if we don't have one
            if conv_id not in self._current_turn:
                turn_id = len(metrics.turns) + 1
                self._current_turn[conv_id] = TurnMetrics(turn_id=turn_id)
            self._current_turn[conv_id].user_speech_start_ms = ts
            # Check for overlap
            self._check_overlap(bot_id, conv_id, ts)
        
        elif event_type == EventType.USER_SPEECH_END:
            speaking["user"] = False
            if conv_id in self._current_turn:
                self._current_turn[conv_id].user_speech_end_ms = ts
            # Check if overlap ended
            self._check_overlap_end(bot_id, conv_id, ts)
        
        # Handle audio sent to backend
        elif event_type == EventType.AUDIO_SENT_TO_BACKEND:
            if conv_id in self._current_turn:
                # Record first audio sent timestamp for this turn
                if self._current_turn[conv_id].audio_sent_ms is None:
                    self._current_turn[conv_id].audio_sent_ms = ts
        
        # Handle transcription
        elif event_type == EventType.TRANSCRIPTION_RECEIVED:
            if conv_id in self._current_turn:
                self._current_turn[conv_id].transcription_received_ms = ts
        
        # Handle bot audio events
        elif event_type == EventType.FIRST_AUDIO_BYTE:
            if conv_id in self._current_turn:
                self._current_turn[conv_id].first_audio_byte_ms = ts
        
        elif event_type == EventType.BOT_SPEECH_START:
            speaking["bot"] = True
            if conv_id in self._current_turn:
                self._current_turn[conv_id].bot_speech_start_ms = ts
            # Check for overlap
            self._check_overlap(bot_id, conv_id, ts)
        
        elif event_type == EventType.BOT_SPEECH_END:
            speaking["bot"] = False
            # Check if overlap ended (after clearing bot speaking state)
            self._check_overlap_end(bot_id, conv_id, ts)
            if conv_id in self._current_turn:
                self._current_turn[conv_id].bot_speech_end_ms = ts
                # Finalize the turn
                self._finalize_turn(conv_id)
        
        # Handle interruption
        elif event_type == EventType.INTERRUPTION:
            if conv_id in self._current_turn:
                self._current_turn[conv_id].was_interrupted = True
    
    def _check_overlap(self, bot_id: str, conv_id: str, ts: float) -> None:
        """Check if both user and bot are speaking (overlap).
        
        Called with lock held. Only emits OVERLAP_START when transitioning
        from non-overlapping to overlapping state.
        """
        speaking = self._speaking_state.get(bot_id, {})
        both_speaking = speaking.get("user") and speaking.get("bot")
        already_overlapping = speaking.get("overlap_active", False)
        
        if both_speaking and not already_overlapping:
            # Starting a new overlap
            speaking["overlap_active"] = True
            speaking["overlap_start_ts"] = ts
            
            if conv_id in self._current_turn:
                self._current_turn[conv_id].had_overlap = True
            
            # Create and persist overlap start event
            overlap_event = ConversationEvent(
                timestamp_ms=ts,
                event_type=EventType.OVERLAP_START,
                conversation_id=conv_id,
                bot_id=bot_id,
            )
            self._events[conv_id].append(overlap_event)
            
            # Persist to database (will be done outside lock)
            self._pending_overlap_events.append(overlap_event)
            
            logger.warning(f"Overlap started in conversation {conv_id}")
    
    def _check_overlap_end(self, bot_id: str, conv_id: str, ts: float) -> None:
        """Check if an active overlap has ended.
        
        Called with lock held when either user or bot stops speaking.
        """
        speaking = self._speaking_state.get(bot_id, {})
        both_speaking = speaking.get("user") and speaking.get("bot")
        was_overlapping = speaking.get("overlap_active", False)
        
        if was_overlapping and not both_speaking:
            # Overlap has ended
            speaking["overlap_active"] = False
            overlap_start_ts = speaking.get("overlap_start_ts")
            speaking["overlap_start_ts"] = None
            
            # Calculate overlap duration
            duration_ms = (ts - overlap_start_ts) if overlap_start_ts else 0
            
            # Update current turn's overlap duration
            if conv_id in self._current_turn:
                self._current_turn[conv_id].overlap_duration_ms += duration_ms
            
            # Create and persist overlap end event
            overlap_event = ConversationEvent(
                timestamp_ms=ts,
                event_type=EventType.OVERLAP_END,
                conversation_id=conv_id,
                bot_id=bot_id,
                metadata={"duration_ms": duration_ms}
            )
            self._events[conv_id].append(overlap_event)
            
            # Persist to database (will be done outside lock)
            self._pending_overlap_events.append(overlap_event)
            
            logger.info(f"Overlap ended in conversation {conv_id} (duration: {duration_ms:.0f}ms)")
    
    def _finalize_turn(self, conv_id: str) -> None:
        """Finalize the current turn and add to metrics.
        
        Called with lock held.
        """
        if conv_id not in self._current_turn:
            return
        
        turn = self._current_turn.pop(conv_id)
        turn.calculate_metrics()
        
        metrics = self._metrics.get(conv_id)
        if metrics:
            metrics.turns.append(turn)
            
            # Persist turn and updated conversation to database
            try:
                self._db.save_turn(conv_id, turn)
                self._db.save_conversation(metrics)
            except Exception as e:
                logger.error(f"Error saving turn to database: {e}")
        
        # Log turn summary
        if turn.turn_around_time_ms is not None:
            ttfb_str = f", TTFB={turn.time_to_first_byte_ms:.0f}ms" if turn.time_to_first_byte_ms else ""
            logger.info(f"Turn {turn.turn_id} complete: TAT={turn.turn_around_time_ms:.0f}ms{ttfb_str}")
    
    def _log_event(self, event: ConversationEvent) -> None:
        """Log event as structured JSON for analysis."""
        # Use a specific logger format for easy parsing
        log_data = {
            "type": "instrumentation_event",
            "event": event.to_dict(),
        }
        logger.debug(f"METRICS: {json.dumps(log_data)}")
    
    def get_events(self, conversation_id: str) -> list[ConversationEvent]:
        """Get all events for a conversation.
        
        Loads from database if not in memory cache.
        """
        with self._lock:
            if conversation_id in self._events and self._events[conversation_id]:
                return list(self._events[conversation_id])
        
        # Try loading from database
        try:
            events = self._db.load_events(conversation_id)
            if events:
                with self._lock:
                    self._events[conversation_id] = events
                return events
        except Exception as e:
            logger.error(f"Error loading events from database: {e}")
        
        return []
    
    def get_metrics(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for a conversation.
        
        Loads from database if not in memory cache.
        """
        with self._lock:
            metrics = self._metrics.get(conversation_id)
            if metrics is not None:
                return metrics
        
        # Try loading from database
        try:
            metrics = self._db.load_conversation(conversation_id)
            if metrics:
                with self._lock:
                    self._metrics[conversation_id] = metrics
                return metrics
        except Exception as e:
            logger.error(f"Error loading metrics from database: {e}")
        
        return None
    
    def get_all_conversations(self) -> list[str]:
        """Get list of all conversation IDs.
        
        Includes both in-memory and database conversations.
        """
        # Get IDs from database (authoritative source)
        try:
            db_ids = set(self._db.load_all_conversation_ids())
        except Exception as e:
            logger.error(f"Error loading conversation IDs: {e}")
            db_ids = set()
        
        with self._lock:
            memory_ids = set(self._metrics.keys())
        
        return list(db_ids | memory_ids)
    
    def get_bot_metrics(self, bot_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for the most recent conversation of a bot.
        
        Checks database for historical conversations.
        """
        # First check in-memory for active conversations
        with self._lock:
            for conv_id, metrics in reversed(list(self._metrics.items())):
                if metrics is not None and metrics.bot_id == bot_id:
                    return metrics
        
        # Check database for historical conversations
        try:
            conv_ids = self._db.load_bot_conversations(bot_id)
            if conv_ids:
                # Load the most recent one
                metrics = self._db.load_conversation(conv_ids[0])
                if metrics:
                    with self._lock:
                        self._metrics[conv_ids[0]] = metrics
                    return metrics
        except Exception as e:
            logger.error(f"Error loading bot metrics from database: {e}")
        
        return None
    
    def export_all(self) -> dict:
        """Export all metrics data for analysis.
        
        Includes both in-memory and persisted data.
        """
        # Get all conversation IDs
        all_conv_ids = self.get_all_conversations()
        
        # Load metrics for each conversation
        conversations = {}
        for conv_id in all_conv_ids:
            metrics = self.get_metrics(conv_id)
            if metrics:
                conversations[conv_id] = metrics.to_full_dict()
        
        # Get database stats
        try:
            db_stats = self._db.get_stats()
        except Exception:
            db_stats = {}
        
        return {
            "conversations": conversations,
            "total_conversations": len(conversations),
            "export_timestamp_ms": time.time() * 1000,
            "database": db_stats,
        }
    
    def get_database_stats(self) -> dict:
        """Get database statistics."""
        try:
            return self._db.get_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}


class MetricsCollector:
    """High-level interface for recording instrumentation events.
    
    This is the main class that should be used throughout the application
    to record events. It provides convenient methods for each event type
    and handles timestamp generation.
    
    The collector can be enabled/disabled at runtime via the `enabled` property.
    When disabled, no events are recorded but API calls still succeed silently.
    
    Usage:
        collector = get_metrics_collector()
        collector.record_user_speech_start(bot_id, conversation_id)
        # ... later ...
        collector.record_bot_speech_end(bot_id, conversation_id)
        
        # Disable collection
        collector.enabled = False
    """
    
    def __init__(self, store: Optional[ConversationStore] = None):
        self.store = store or ConversationStore()
        self._enabled = True  # Metrics collection enabled by default
    
    @property
    def enabled(self) -> bool:
        """Whether metrics collection is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable metrics collection."""
        self._enabled = bool(value)
        logger.info(f"Metrics collection {'enabled' if self._enabled else 'disabled'}")
    
    def _now_ms(self) -> float:
        """Get current timestamp in milliseconds."""
        return time.time() * 1000
    
    def _record(
        self,
        event_type: EventType,
        bot_id: str,
        conversation_id: str,
        turn_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[ConversationEvent]:
        """Record an event with the given parameters.
        
        Returns None if metrics collection is disabled.
        """
        if not self._enabled:
            return None
        
        event = ConversationEvent(
            timestamp_ms=self._now_ms(),
            event_type=event_type,
            conversation_id=conversation_id,
            bot_id=bot_id,
            turn_id=turn_id,
            metadata=metadata or {},
        )
        self.store.record_event(event)
        return event
    
    def record_event(self, event: ConversationEvent) -> None:
        """Record a pre-built event directly.
        
        This is useful when receiving events from external sources (like browser SDK)
        that already have timestamps and metadata populated.
        
        Does nothing if metrics collection is disabled.
        """
        if not self._enabled:
            return
        self.store.record_event(event)
    
    # Conversation lifecycle
    
    def record_conversation_start(
        self,
        bot_id: str,
        conversation_id: str,
        backend_type: str = "elevenlabs",
    ) -> ConversationEvent:
        """Record conversation start."""
        return self._record(
            EventType.CONVERSATION_START,
            bot_id,
            conversation_id,
            metadata={"backend_type": backend_type},
        )
    
    def record_conversation_end(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record conversation end."""
        return self._record(EventType.CONVERSATION_END, bot_id, conversation_id)
    
    # User speech events
    
    def record_user_speech_start(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when user starts speaking."""
        return self._record(EventType.USER_SPEECH_START, bot_id, conversation_id)
    
    def record_user_speech_end(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when user stops speaking."""
        return self._record(EventType.USER_SPEECH_END, bot_id, conversation_id)
    
    def record_user_audio_received(
        self,
        bot_id: str,
        conversation_id: str,
        audio_bytes: int = 0,
    ) -> ConversationEvent:
        """Record when user audio is received from meeting."""
        return self._record(
            EventType.USER_AUDIO_RECEIVED,
            bot_id,
            conversation_id,
            metadata={"audio_bytes": audio_bytes},
        )
    
    # Backend communication events
    
    def record_audio_sent_to_backend(
        self,
        bot_id: str,
        conversation_id: str,
        audio_bytes: int = 0,
    ) -> ConversationEvent:
        """Record when audio is sent to voice AI backend."""
        return self._record(
            EventType.AUDIO_SENT_TO_BACKEND,
            bot_id,
            conversation_id,
            metadata={"audio_bytes": audio_bytes},
        )
    
    def record_transcription_received(
        self,
        bot_id: str,
        conversation_id: str,
        text: str = "",
    ) -> ConversationEvent:
        """Record when transcription is received from backend."""
        return self._record(
            EventType.TRANSCRIPTION_RECEIVED,
            bot_id,
            conversation_id,
            metadata={"text": text[:100]},  # Truncate for logging
        )
    
    def record_agent_response_start(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when agent starts generating response."""
        return self._record(EventType.AGENT_RESPONSE_START, bot_id, conversation_id)
    
    def record_agent_response_text(
        self,
        bot_id: str,
        conversation_id: str,
        text: str = "",
    ) -> ConversationEvent:
        """Record agent response text."""
        return self._record(
            EventType.AGENT_RESPONSE_TEXT,
            bot_id,
            conversation_id,
            metadata={"text": text[:100]},
        )
    
    # Bot audio events
    
    def record_first_audio_byte(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when first audio byte is received from backend."""
        return self._record(EventType.FIRST_AUDIO_BYTE, bot_id, conversation_id)
    
    def record_bot_speech_start(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when bot starts speaking (audio playing)."""
        return self._record(EventType.BOT_SPEECH_START, bot_id, conversation_id)
    
    def record_bot_speech_end(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when bot finishes speaking."""
        return self._record(EventType.BOT_SPEECH_END, bot_id, conversation_id)
    
    def record_audio_sent_to_meeting(
        self,
        bot_id: str,
        conversation_id: str,
        audio_bytes: int = 0,
    ) -> ConversationEvent:
        """Record when audio is sent back to the meeting."""
        return self._record(
            EventType.AUDIO_SENT_TO_MEETING,
            bot_id,
            conversation_id,
            metadata={"audio_bytes": audio_bytes},
        )
    
    # Conversation flow events
    
    def record_interruption(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record when user interrupts the bot."""
        return self._record(EventType.INTERRUPTION, bot_id, conversation_id)
    
    # Pipecat-specific events (for future use)
    
    def record_stt_start(
        self,
        bot_id: str,
        conversation_id: str,
        provider: str = "",
    ) -> ConversationEvent:
        """Record STT processing start (Pipecat only)."""
        return self._record(
            EventType.STT_START,
            bot_id,
            conversation_id,
            metadata={"provider": provider},
        )
    
    def record_stt_end(
        self,
        bot_id: str,
        conversation_id: str,
        text: str = "",
    ) -> ConversationEvent:
        """Record STT processing end (Pipecat only)."""
        return self._record(
            EventType.STT_END,
            bot_id,
            conversation_id,
            metadata={"text": text[:100]},
        )
    
    def record_llm_start(
        self,
        bot_id: str,
        conversation_id: str,
        provider: str = "",
        model: str = "",
    ) -> ConversationEvent:
        """Record LLM processing start (Pipecat only)."""
        return self._record(
            EventType.LLM_START,
            bot_id,
            conversation_id,
            metadata={"provider": provider, "model": model},
        )
    
    def record_llm_end(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record LLM processing end (Pipecat only)."""
        return self._record(EventType.LLM_END, bot_id, conversation_id)
    
    def record_tts_start(
        self,
        bot_id: str,
        conversation_id: str,
        provider: str = "",
    ) -> ConversationEvent:
        """Record TTS processing start (Pipecat only)."""
        return self._record(
            EventType.TTS_START,
            bot_id,
            conversation_id,
            metadata={"provider": provider},
        )
    
    def record_tts_end(
        self,
        bot_id: str,
        conversation_id: str,
    ) -> ConversationEvent:
        """Record TTS processing end (Pipecat only)."""
        return self._record(EventType.TTS_END, bot_id, conversation_id)
    
    # Query methods
    
    def get_conversation_metrics(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for a specific conversation."""
        return self.store.get_metrics(conversation_id)
    
    def get_bot_metrics(self, bot_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for a bot's most recent conversation."""
        return self.store.get_bot_metrics(bot_id)
    
    def get_conversation_events(self, conversation_id: str) -> list[ConversationEvent]:
        """Get all events for a conversation."""
        return self.store.get_events(conversation_id)
    
    def get_all_conversations(self) -> list[str]:
        """Get list of all conversation IDs."""
        return self.store.get_all_conversations()
    
    def export_all(self) -> dict:
        """Export all metrics data."""
        return self.store.export_all()
    
    def subscribe_to_events(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Subscribe to receive all events in real-time."""
        self.store.subscribe(callback)
    
    def unsubscribe_from_events(self, callback: Callable[[ConversationEvent], None]) -> None:
        """Unsubscribe from event notifications."""
        self.store.unsubscribe(callback)


# Global singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global MetricsCollector instance.
    
    Returns:
        The singleton MetricsCollector instance.
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global MetricsCollector (mainly for testing)."""
    global _metrics_collector
    _metrics_collector = None


# Convenience singleton for direct import: `from core.instrumentation import metrics_collector`
metrics_collector = get_metrics_collector()
