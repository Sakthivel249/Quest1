import sqlite3
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.db")

def init_db():
    """Initialize the SQLite database and create the history table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            dialogue TEXT NOT NULL,
            mode TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_search(url: str, dialogue: str, mode: str, frame_idx: int, timestamp: str, image_path: str):
    """Save a successful search to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (url, dialogue, mode, frame_idx, timestamp, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (url, dialogue, mode, frame_idx, timestamp, image_path))
        conn.commit()
        conn.close()
        logger.info("Search saved to history database.")
    except Exception as e:
        logger.error(f"Failed to save search history: {e}")

def get_history():
    """Retrieve all search history ordered by newest first."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT url, dialogue, mode, timestamp, image_path, created_at 
            FROM history 
            ORDER BY id DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []

# Ensure DB is initialized when module is imported
init_db()
