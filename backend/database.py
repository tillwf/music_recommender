import sqlite3
from datetime import datetime

from config import DATABASE_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    spotify_track_id TEXT NOT NULL,
    artist_name      TEXT NOT NULL,
    track_name       TEXT NOT NULL,
    known            BOOLEAN NOT NULL,
    liked            BOOLEAN NOT NULL,
    feedback         TEXT,
    strategy_text    TEXT,
    llm_prompt       TEXT,
    llm_response     TEXT,
    llm_model        TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect():
    return sqlite3.connect(DATABASE_PATH)


def init_db():
    with _connect() as conn:
        conn.execute(_SCHEMA)


def add_vote(session_id, track_id, artist, track, known, liked,
             feedback=None, strategy_text=None, llm_prompt=None, llm_response=None, llm_model=None):
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO votes (session_id, spotify_track_id, artist_name, track_name,
                               known, liked, feedback, strategy_text, llm_prompt, llm_response, llm_model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, track_id, artist, track, known, liked,
             feedback, strategy_text, llm_prompt, llm_response, llm_model, now),
        )


def get_session_history(session_id):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM votes WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_summary_data(session_id):
    """Format session history for the summarizer LLM."""
    history = get_session_history(session_id)
    lines = []
    for i, v in enumerate(history, 1):
        known_str = "knew" if v["known"] else "didn't know"
        liked_str = "liked" if v["liked"] else "didn't like"
        lines.append(f"{i}. {v['artist_name']} - {v['track_name']} ({known_str}, {liked_str})")
    return "\n".join(lines)
