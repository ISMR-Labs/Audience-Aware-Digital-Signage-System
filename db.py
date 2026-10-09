"""Tiny SQLite wrapper (thread-safe via a lock, WAL mode)."""

import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    color TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS ads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    filename TEXT NOT NULL,
    type TEXT NOT NULL,              -- image | video
    duration REAL NOT NULL,          -- seconds on screen
    enabled INTEGER DEFAULT 1,
    active_from INTEGER DEFAULT 0,   -- hour of day (inclusive)
    active_to INTEGER DEFAULT 24,    -- hour of day (exclusive)
    created_at REAL
);
CREATE TABLE IF NOT EXISTS impressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ad_id INTEGER REFERENCES ads(id) ON DELETE CASCADE,
    category_id INTEGER,
    started_at REAL,
    ended_at REAL,
    viewers INTEGER,      -- distinct people present during the ad
    lookers INTEGER,      -- of those, how many looked at the screen
    avg_people REAL,
    max_people INTEGER,
    frames INTEGER,
    reason TEXT           -- timer | skip | stop
);
CREATE INDEX IF NOT EXISTS idx_imp_ended ON impressions(ended_at);
CREATE INDEX IF NOT EXISTS idx_imp_ad ON impressions(ad_id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

DEFAULT_SETTINGS = {
    "default_duration": "10",     # seconds for image ads
    "half_life_hours": "2",       # decay of old evidence in the bandit
    "epsilon": "0.1",             # probability of a purely random category (exploration floor)
    "min_present_s": "0.5",       # a person must be in view this long to count as a viewer
    "min_look_s": "0.7",          # ... and face the screen this long to count as a looker
    "attention_angle": "25",      # degrees, from people_counter
    "running": "0",
}


class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript("PRAGMA journal_mode=WAL;")
            self._migrate_autoincrement()
            self.conn.executescript("PRAGMA foreign_keys=ON;" + SCHEMA)
            for k, v in DEFAULT_SETTINGS.items():
                self.conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))
            self.conn.commit()

    def _migrate_autoincrement(self):
        """Early databases used plain INTEGER PRIMARY KEY, which lets SQLite reuse the id of a deleted ad;
        a later upload could then inherit the old ad's play history. Rebuild such tables with AUTOINCREMENT."""
        for table in ("categories", "ads", "impressions"):
            row = self.conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if row is None or "AUTOINCREMENT" in row[0]:
                continue
            create = SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table} (")[1].split(");")[0]
            self.conn.executescript(
                f"PRAGMA foreign_keys=OFF; BEGIN;"
                f"CREATE TABLE {table}__new ({create});"
                f"INSERT INTO {table}__new SELECT * FROM {table};"
                f"DROP TABLE {table}; ALTER TABLE {table}__new RENAME TO {table}; COMMIT;"
            )
            print(f"[db] migrated {table} to AUTOINCREMENT ids", flush=True)

    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def exec(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # settings -------------------------------------------------------------
    def setting(self, key, cast=str):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        val = row["value"] if row else DEFAULT_SETTINGS.get(key)
        return cast(val) if val is not None else None

    def set_setting(self, key, value):
        self.exec("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (key, str(value)))

    def all_settings(self):
        return {r["key"]: r["value"] for r in self.q("SELECT key, value FROM settings")}

    # convenience ----------------------------------------------------------
    def add_category(self, name, color=None):
        return self.exec("INSERT INTO categories(name, color, created_at) VALUES (?, ?, ?)",
                         (name.strip(), color, time.time()))

    def add_ad(self, category_id, name, filename, type_, duration):
        return self.exec(
            "INSERT INTO ads(category_id, name, filename, type, duration, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (category_id, name, filename, type_, duration, time.time()),
        )

    def add_impression(self, ad_id, category_id, started_at, ended_at, m, reason):
        return self.exec(
            "INSERT INTO impressions(ad_id, category_id, started_at, ended_at, viewers, lookers, avg_people, "
            "max_people, frames, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ad_id, category_id, started_at, ended_at, m["viewers"], m["lookers"], m["avg_people"],
             m["max_people"], m["frames"], reason),
        )
