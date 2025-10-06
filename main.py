# app.py
# ------------------------------------------------------------
# PG Log Search & Clustering (Qdrant + OpenAI) — Fast Indexing + Deadlock Aware
# - FAST INDEX (template-only + dedup) with persistent embedding cache (SQLite)
# - Severity filter at ingest
# - OpenAI embeddings w/ retries/backoff/adaptive batching
# - gRPC-safe UUIDv5 IDs
# - Sharded upserts + small batched Qdrant writes (with retries)
# - Indexing only on demand (button / file change)
# - Auto-migration for legacy table ("timestamp" -> ts)
# - Reset collections + Smoke test in sidebar
# - Robust vector-name detection for mixed qdrant-client versions
# - Deadlock-aware search path: extracts PIDs, pulls ±120s context, prints exact SQL
# ------------------------------------------------------------

import os
import re
import json
import time
import math
import uuid
import random
import sqlite3
import tempfile
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from dateutil import parser as dtparser

import pandas as pd
import plotly.express as px
import dateutil.parser as dtparser
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import psycopg2
import requests
import streamlit as st
import textwrap
from sqlalchemy import create_engine
from openai import OpenAI
  
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
)

from sklearn.cluster import KMeans
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

# -------------------------
# Streamlit page + session
# -------------------------
st.set_page_config(page_title="Askpglog", layout="wide")

# -------------------------
# PostgreSQL settings
PG_HOST = "localhost"
PG_DB = "tsdb"
PG_USER = "postgres"
PG_PASSWORD = "accesspass"
PG_PORT = 5433
TABLE_NAME = "uploaded_logs"

# Embeddings
EMBED_BACKEND = "openai"  # "openai" or "st"
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# Optional ST fallback (only if EMBED_BACKEND="st")
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = None

# OpenAI embed tuning
OPENAI_TIMEOUT = 120
OPENAI_MAX_RETRIES = 6
OPENAI_BATCH = 128            # bump default for speed
OPENAI_MIN_BATCH = 16         # safe floor
OPENAI_BACKOFF_BASE = 1.8

# Fast indexing + cache
FAST_INDEX = True
CACHE_EMB = True
EMB_CACHE_PATH = "emb_cache.sqlite"  # persisted across runs

# Qdrant
COLLECTION = "pg_logs"

QDRANT_URL = None
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_GRPC = True
QDRANT_GRPC_PORT = 6334
QDRANT_TIMEOUT = 180           # bump default for long writes
QDRANT_BATCH = 1000            # bigger batches = fewer calls
QDRANT_MAX_RETRIES = 6
QDRANT_BACKOFF_BASE = 1.8
USE_SPARSE = False
QDRANT_RESET = False

# Shard size
SHARD_SIZE = 3000

# --- ADD THIS BLOCK NEAR THE TOP OF YOUR SCRIPT ---
# Initialize session state for Qdrant reset confirmation (ticked by default)
if "qdrant_reset_confirmed" not in st.session_state:
    st.session_state.qdrant_reset_confirmed = True

# Initialize a dynamic key for the file uploader to enable resetting
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0 
# --------------------------------------------------

# -------------------------
# UTIL: Wait for Qdrant (HTTP ping)
# -------------------------
def wait_for_qdrant(host: str, port: int, timeout: int = 30):
    url = f"http://{host}:{port}/collections"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Qdrant not ready after waiting.")

wait_for_qdrant(QDRANT_HOST, QDRANT_PORT)
client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# -------------------------
# DB helpers & migration
# -------------------------
def connect_pg():
    return psycopg2.connect(
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD, host=PG_HOST, port=PG_PORT
    )

def ensure_table_exists():
    TABLE_NAME = "uploaded_logs"
    conn = connect_pg()
    with conn.cursor() as cur:
        # 1️⃣ Create table if it doesn't exist
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                raw_line   TEXT,
                ts         TIMESTAMPTZ,
                tz         TEXT,
                pid        INTEGER,
                ident      TEXT,
                host_ip    TEXT,
                port       INTEGER,
                level      TEXT,
                message    TEXT,
                detail     TEXT,
                statement  TEXT
            )
        """)
        conn.commit()

        # 2️⃣ Get existing columns
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
        """, (TABLE_NAME,))
        cols = {r[0] for r in cur.fetchall()}

        # 3️⃣ Handle old "timestamp" column if it exists
        if "timestamp" in cols and "ts" not in cols:
            cur.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS ts TIMESTAMPTZ')
            cur.execute(f'UPDATE {TABLE_NAME} SET ts = "timestamp"::timestamptz WHERE ts IS NULL')
            cur.execute(f'ALTER TABLE {TABLE_NAME} DROP COLUMN "timestamp"')
            conn.commit()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
            """, (TABLE_NAME,))
            cols = {r[0] for r in cur.fetchall()}

        # 4️⃣ Ensure all desired columns exist
        wanted = {
            "raw_line":"TEXT","tz":"TEXT","ident":"TEXT","host_ip":"TEXT","port":"INTEGER",
            "detail":"TEXT","statement":"TEXT","level":"TEXT","message":"TEXT","pid":"INTEGER","ts":"TIMESTAMPTZ"
        }
        for cname, ctype in wanted.items():
            if cname not in cols:
                cur.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN {cname} {ctype}')
        conn.commit()

        # 5️⃣ Create unique index if not exists
        cur.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname = '{TABLE_NAME}_uq_ts_pid_msg'
                ) THEN
                    EXECUTE 'CREATE UNIQUE INDEX {TABLE_NAME}_uq_ts_pid_msg
                             ON {TABLE_NAME} (ts, pid, COALESCE(message, ''''))';
                END IF;
            END$$;
        """)
        conn.commit()

    conn.close()

def insert_events_pg(events):
    if not events:
        return 0
    conn = connect_pg()
    with conn.cursor() as cur:
        inserted = 0
        for e in events:
            cur.execute(
                f"SELECT 1 FROM {TABLE_NAME} WHERE ts=%s AND pid=%s AND message=%s LIMIT 1",
                (e["ts"], e.get("pid"), e.get("message"))
            )
            if not cur.fetchone():
                cur.execute(
                    f"""INSERT INTO {TABLE_NAME}
                        (raw_line, ts, tz, pid, ident, host_ip, port, level, message, detail, statement)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        e.get("raw", ""),
                        e["ts"],
                        e.get("tz"),
                        e.get("pid"),
                        e.get("ident"),
                        e.get("host_ip"),
                        e.get("port"),
                        e.get("level"),
                        e.get("message"),
                        e.get("detail"),
                        e.get("statement"),
                    )
                )
                inserted += 1
        conn.commit()
    conn.close()
    return inserted

def load_events_from_db():
    conn = connect_pg()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT ts, tz, pid, ident, host_ip, port, level, message, detail, statement
            FROM {TABLE_NAME}
            ORDER BY ts ASC
        """)
        rows = cur.fetchall()
    conn.close()
    events = []
    for ts, tz, pid, ident, host_ip, port, level, message, detail, statement in rows:
        events.append({
            "ts": ts, "tz": tz, "pid": pid, "ident": ident,
            "host_ip": host_ip, "port": port, "level": level,
            "message": message, "detail": detail, "statement": statement
        })
    return events

# -------------------------
# Embedding cache (SQLite)
# -------------------------
def _emb_db():
    conn = sqlite3.connect(EMB_CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emb_cache (
            key TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            model TEXT NOT NULL,
            text TEXT NOT NULL,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn

def _vec_to_blob(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float32:
        vec = vec.astype("float32")
    return vec.tobytes(order="C")

def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        raise ValueError("Cached vector dim mismatch")
    return arr

def _embed_cache_get(texts, backend, model, dim_hint=None):
    if not CACHE_EMB:
        return {}, set(range(len(texts)))
    conn = _emb_db()
    cur = conn.cursor()
    got = {}
    missing_idx = set()
    for i, t in enumerate(texts):
        key = hashlib.sha256((backend + "::" + model + "::" + t).encode("utf-8")).hexdigest()
        row = cur.execute("SELECT dim, vec FROM emb_cache WHERE key=?", (key,)).fetchone()
        if row:
            dim, blob = row
            vec = _blob_to_vec(blob, dim)
            got[i] = vec
        else:
            missing_idx.add(i)
    conn.close()
    return got, missing_idx

def _embed_cache_put(texts, vectors, backend, model):
    if not CACHE_EMB or not len(texts):
        return
    dim = int(vectors.shape[1])
    conn = _emb_db()
    cur = conn.cursor()
    rows = []
    for t, v in zip(texts, vectors):
        key = hashlib.sha256((backend + "::" + model + "::" + t).encode("utf-8")).hexdigest()
        blob = _vec_to_blob(np.asarray(v, dtype="float32"))
        rows.append((key, backend, model, t, dim, blob))
    cur.executemany("INSERT OR REPLACE INTO emb_cache (key, backend, model, text, dim, vec) VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

# -------------------------
# OpenAI embeddings + fallback
# -------------------------
def _openai_post_embeddings(inputs: list[str]) -> list[list[float]]:
    assert OPENAI_API_KEY, "OPENAI_API_KEY must be set"
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url, headers=headers,
                json={"model": OPENAI_EMBED_MODEL, "input": inputs},
                timeout=OPENAI_TIMEOUT,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"OpenAI transient {resp.status_code}: {resp.text[:180]}")
            resp.raise_for_status()
            data = resp.json()["data"]
            return [d["embedding"] for d in data]
        except (requests.ReadTimeout, requests.ConnectTimeout, requests.HTTPError, requests.ConnectionError) as e:
            last_err = e
            sleep_s = (OPENAI_BACKOFF_BASE ** attempt) + random.uniform(0, 0.75)
            sleep_s = min(sleep_s, 30.0)
            try: st.warning(f"Embedding retry {attempt}/{OPENAI_MAX_RETRIES} in ~{sleep_s:.1f}s ({type(e).__name__})")
            except Exception: pass
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI embeddings failed after {OPENAI_MAX_RETRIES} retries: {last_err}")

def _openai_embed_all(texts: list[str]) -> np.ndarray:
    # cache first
    cached, missing_idx = _embed_cache_get(texts, "openai", OPENAI_EMBED_MODEL)
    vectors = [None] * len(texts)
    for i, v in cached.items():
        vectors[i] = v

    # embed misses in adaptive batches
    def embed_batch(batch_texts):
        vecs = _openai_post_embeddings(batch_texts)
        return np.asarray(vecs, dtype="float32")

    i = 0
    batch = max(OPENAI_MIN_BATCH, OPENAI_BATCH)
    misses = [idx for idx in sorted(missing_idx)]
    while i < len(misses):
        sl = misses[i:i+batch]
        chunk = [texts[j] for j in sl]
        try:
            vecs = embed_batch(chunk)
            _embed_cache_put(chunk, vecs, "openai", OPENAI_EMBED_MODEL)
            for j, v in zip(sl, vecs):
                vectors[j] = v
            i += batch
        except RuntimeError:
            if batch > OPENAI_MIN_BATCH:
                batch = max(OPENAI_MIN_BATCH, batch // 2)
                try: st.info(f"Reducing OpenAI batch size to {batch} due to timeouts/errors.")
                except Exception: pass
                continue
            raise

    return np.vstack([np.asarray(v, dtype="float32") for v in vectors])

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    if EMBED_BACKEND == "openai":
        return _openai_embed_all(texts)
    # local fallback
    global embedder
    try:
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
        # cache for ST backend too
        cached, missing_idx = _embed_cache_get(texts, "st", EMBEDDER_MODEL)
        vectors = [None]*len(texts)
        for i, v in cached.items(): vectors[i] = v
        misses = [idx for idx in sorted(missing_idx)]
        if misses:
            vecs = embedder.encode([texts[i] for i in misses], show_progress_bar=False, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype="float32")
            _embed_cache_put([texts[i] for i in misses], vecs, "st", EMBEDDER_MODEL)
            for i_m, v in zip(misses, vecs): vectors[i_m] = v
        return np.vstack([np.asarray(v, dtype="float32") for v in vectors])
    except Exception as e:
        st.error(f"SentenceTransformers failed: {e}")
        raise

def call_openai_chat(messages, temperature=0, max_tokens=1200):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for chat.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_CHAT_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# -------------------------
# Parsing
# -------------------------
PREFIX = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)\s+(?P<tz>[A-Z]+)\s+\[(?P<pid>\d+)\]'
    r'(?:\s+(?P<ident>\S+@\S+)\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\((?P<port>\d+)\))?\s+(?P<rest>.*)$'
)
LEVEL_PREFIX = re.compile(r'^(?P<level>LOG|ERROR|WARNING|FATAL|PANIC):\s{2}(?P<msg>.*)$')
DETAIL_LINE = re.compile(r'^DETAIL:\s{2}(?P<detail>.*)$')
STATEMENT_LINE = re.compile(r'^STATEMENT:\s{2}(?P<stmt>.*)$')

def parse_events(text: str):
    events = []
    open_event_by_pid = {}
    def close(pid):
        ev = open_event_by_pid.pop(pid, None)
        if ev: events.append(ev)

    for raw in text.splitlines():
        raw = raw.rstrip("\n")
        m = PREFIX.match(raw)
        if not m:
            continue
        ts_str = m.group("ts"); tz = m.group("tz"); pid = int(m.group("pid"))
        ident = m.group("ident"); ip = m.group("ip")
        port = int(m.group("port")) if m.group("port") else None
        rest = m.group("rest") or ""
        md = DETAIL_LINE.match(rest); ms = STATEMENT_LINE.match(rest); ml = LEVEL_PREFIX.match(rest)

        try:
            fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in ts_str else "%Y-%m-%d %H:%M:%S"
            ts = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            ts = dtparser.parse(ts_str)
            if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)

        if ml:
            close(pid)
            level = ml.group("level"); msg = ml.group("msg")
            open_event_by_pid[pid] = {
                "raw": raw, "ts": ts, "tz": tz, "pid": pid, "ident": ident,
                "host_ip": ip, "port": port, "level": level, "message": msg,
                "detail": None, "statement": None,
            }
        elif md:
            ev = open_event_by_pid.get(pid)
            if ev: ev["detail"] = (ev.get("detail") + " " if ev.get("detail") else "") + md.group("detail")
        elif ms:
            ev = open_event_by_pid.get(pid)
            if ev: ev["statement"] = (ev.get("statement") + " " if ev.get("statement") else "") + ms.group("stmt")
        else:
            close(pid)
            open_event_by_pid[pid] = {
                "raw": raw, "ts": ts, "tz": tz, "pid": pid, "ident": ident,
                "host_ip": ip, "port": port, "level": "LOG", "message": rest,
                "detail": None, "statement": None,
            }
    for pid in list(open_event_by_pid.keys()):
        close(pid)
    return events

# -------------------------
# Text building & fingerprinting
# -------------------------
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
IPV4 = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
NUM = re.compile(r'\b\d+\b')
QUOTED = re.compile(r"'[^']*'")

def build_full_text(e):
    head = f"{e['ts'].strftime('%Y-%m-%d %H:%M:%S.%f')} {e.get('tz','')} [{e.get('pid','-')}] {e.get('ident','-')} {e.get('host_ip','-')}({e.get('port','-')}) {e.get('level','')}"
    parts = [head]
    if e.get("message"): parts.append(e["message"])
    if e.get("detail"): parts.append(f"DETAIL: {e['detail']}")
    if e.get("statement"): parts.append(f"STATEMENT: {e['statement']}")
    return "\n".join(parts).strip()

def fingerprint_text(text: str) -> str:
    text = EMAIL.sub("<EMAIL>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = IPV4.sub("<IP>", text)
    text = QUOTED.sub("<STR>", text)
    text = NUM.sub("<NUM>", text)
    return text

# -------------------------
# Deterministic UUIDv5 IDs (gRPC-safe)
# -------------------------
def make_point_id(e: dict) -> str:
    basis = f"{int(e['ts'].timestamp())}|{e.get('pid')}|{(e.get('message') or '')[:80]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))

# -------------------------
# Qdrant client & collections
# -------------------------
def make_qdrant_client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT, prefer_grpc=QDRANT_GRPC, grpc_port=QDRANT_GRPC_PORT)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT, prefer_grpc=QDRANT_GRPC, grpc_port=QDRANT_GRPC_PORT)

client = make_qdrant_client()

def qdrant_index_exists():
    try:
        return client.collection_exists(COLLECTION)
    except Exception:
        return False

def recreate_base_collection(dim: int, fast: bool):
    # Delete existing collection if it exists
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    # Prepare vector config
    if fast:
        vectors_config = {"text_tpl": VectorParams(size=dim, distance=Distance.COSINE)}
    else:
        vectors_config = {
            "text_full": VectorParams(size=dim, distance=Distance.COSINE),
            "text_tpl":  VectorParams(size=dim, distance=Distance.COSINE),
        }

    # Create the collection
    if USE_SPARSE:
        if not client.collection_exists(COLLECTION):
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=vectors_config,
                sparse_vectors_config={"text_sparse": SparseVectorParams()}
            )
    else:
        if not client.collection_exists(COLLECTION):
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=vectors_config
            )

st.markdown("""
<style>
/* Fixed tab bar */
div[data-baseweb="tab-list"] {
    position: fixed;
    top: 3rem;
    left: 0;
    right: 0;
    background-color: white;
    z-index: 1000;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #ddd;
    display: flex;
    justify-content: flex-end;
}

/* Tabs padding for content */
.block-container {
    padding-top: 7rem;
}

/* All tabs same blue background + bold black text + border */
div[data-baseweb="tab-list"] [role="tab"] {
    background-color: #cce5ff !important;  
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px 8px 0 0;
    border: 1px solid #888;  /* Normal tabs border */
    padding: 0.5rem 1rem;
    margin-right: 5px;  /* Slight gap between tabs for borders */
    position: relative;
}

/* Selected tab with deep shadow + darker border */
div[data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {
    color: black !important;
    font-weight: bold !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.8), 0 0 30px rgba(0,0,0,0.6);
    border: 2px solid #111;  /* Much darker border for selected tab */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Fixed tab bar background with light grey */
div[data-baseweb="tab-list"] {
    position: fixed;
    top: 3rem;
    left: 0;
    right: 0;
    background-color: #f0f0f0;  /* light grey */
    z-index: 1000;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #ccc;  /* subtle border */
    display: flex;
    justify-content: flex-end;
}

/* Tabs padding for content below */
.block-container {
    padding-top: 7rem;
}

/* Individual tabs text + styling */
div[data-baseweb="tab-list"] [role="tab"] {
    color: black !important;   /* dark text on light grey */
    font-weight: bold;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    margin-right: 0px;
}

/* Highlight selected tab */
div[data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {
    color: black !important;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------- Tabs -------------------
app_tab, about_tab, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 App",
    "ℹ️ About",
    "🧪 Qdrant Smoke Test",
    "🗑️ Reset Qdrant",
    "🗑️ Reset PostgreSQL",
    "⚡ Fast Indexing",
    "🔎 Filters"
])

with about_tab:
    # Add padding from top so content is visible below fixed tabs
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)

    # Wrap content in a slightly darker light grey box with black text
    st.markdown("""
    <div style="
        background-color: #E0E0E0;  /* slightly darker light grey */
        color: #000000;             /* black text */
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #BEBEBE;  /* slightly darker grey border */
        font-size: 15px;             /* increased general font size */
    ">
        <div style="display: flex; gap: 25px;">
            <div style="flex:1">
                <h4 style="font-size:18px; font-weight:bold; color:#000000;">📘 How to Use the App</h4>
                <ul style="font-size:15px; color:#000000;">
                    <li>Upload PostgreSQL log files in the sidebar (.txt, .log).</li>
                    <li>Click <b>Ingest & Index</b> (or enable auto-index) to process logs.</li>
                    <li>Search logs with keywords or paste log lines.</li>
                    <li>Filter results by severity or time range.</li>
                    <li>View results with surrounding context.</li>
                    <li>If deadlock is detected → shows PIDs, SQLs, and wait cycle.</li>
                    <li>Click <b>✨ Generate Summary</b> for root cause and fixes.</li>
                    <li>Use tabs for smoke test, reset options, or fast indexing.</li>
                </ul>
            </div>
            <div style="flex:1">
                <h4 style="font-size:18px; font-weight:bold; color:#000000;">📝 Description</h4>
                <p style="font-size:15px; color:#000000;">
                    Askpglog is a Streamlit app for PostgreSQL log analysis. 
                    It integrates with Qdrant for semantic search and can generate AI summaries. 
                    Built for DB admins and SREs to troubleshoot faster.
                </p>
                <b style="font-size:18px; font-weight:bold; color:#000000;">🚀 Features:</b>
                <ul style="font-size:15px; color:#000000;">
                    <li>Upload and preview logs (multi-format support).</li>
                    <li>Fast indexing with deduplication and caching.</li>
                    <li>Filter logs by severity (ERROR, WARNING, FATAL, PANIC).</li>
                    <li>Reset Qdrant collection and also cache collection if needed.</li>
                    <li>Semantic search using embeddings (OpenAI/ST).</li>
                    <li>Deadlock detection and SQL reporting.</li>
                    <li>AI-based summaries (Root Cause + Fixes).</li>
                    <li>Severity charts and log counts.</li>
                    <li>Reset/Smoke test tools for PostgreSQL and Qdrant.</li>
                </ul>
                <b style="font-size:18px; font-weight:bold; color:#000000;">💡 Use Cases:</b>
                <ul style="font-size:15px; color:#000000;">
                    <li>Debug PostgreSQL errors, warnings, and crashes.</li>
                    <li>Investigate and resolve deadlocks.</li>
                    <li>Monitor recurring log issues.</li>
                    <li>Assist incident response with quick summaries.</li>
                    <li>Prepare incident reports with actual log evidence.</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------- Existing Tabs -------------------
with tab1:
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)
    st.info("Runs a quick check to verify that your Qdrant database is running and functioning as expected")

    if st.button("Run Smoke Test"):
        with st.spinner("Running Qdrant smoke test (create->upsert->search->delete)..."):
            try:
                client_smoke = make_qdrant_client()
                test_col = f"__smoke_test_{int(time.time())}_{random.randint(1000,9999)}"
                # create minimal collection with a single named vector "v"
                client_smoke.create_collection(
                    collection_name=test_col,
                    vectors_config={"v": VectorParams(size=1, distance=Distance.COSINE)}
                )

                pt_id = str(uuid.uuid4())
                pt = PointStruct(id=pt_id, vector={"v":[0.123]}, payload={"ok": True})
                client_smoke.upsert(collection_name=test_col, points=[pt], wait=True)

                # search back the vector
                results = client_smoke.search(collection_name=test_col, query_vector=("v", [0.123]), limit=1)
                found = bool(results)

                # cleanup
                try:
                    client_smoke.delete_collection(test_col)
                except Exception:
                    pass

                if found:
                    st.success("Smoke test passed — Qdrant accepted writes and returned expected results ✅")
                else:
                    st.error("Smoke test failed — write succeeded but search returned no results.")
            except Exception as e:
                st.error(f"Smoke test failed: {e}")
                import traceback
                st.text(traceback.format_exc())

with tab2:
    LOG_COLLECTION = "pg_logs"
    CACHE_COLLECTION = "query_cache"
    EMBEDDING_DIM = 1536 
    
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)
    st.info("⚠️ **Resetting collections is permanent.** Choose the data to wipe.")

    # --- QDRANT CLIENT CONNECTION ---
    try:
        client_reset = make_qdrant_client()
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        # st.stop()

    # ----------------------------------------------------
    # 1. Reset PG Logs Collection (Full App State Reset)
    # ----------------------------------------------------
    st.subheader("Reset Qdrant Collection")
    st.warning(f"This clears all embedded log datas stored in the collection")
    
    confirm_logs = st.checkbox(
        f"I understand — reset the Qdrant collection",
        key="pg_logs_reset_confirmed",
        value=True  # Default state is ticked
    )

    if st.button("🔴 Reset Qdrant logs Collection", key="reset_logs_button") and confirm_logs:
        
        # --- CRITICAL: Clear Streamlit State & File Uploader ---
        st.session_state.uploaded_data = {} 
        st.session_state.upload_sig = None
        st.session_state.index_ready = False
        
        # Increment the key to force the st.file_uploader widget to reset
        if 'file_uploader_key' in st.session_state:
            st.session_state.file_uploader_key += 1
        else:
            st.session_state.file_uploader_key = 1 
        # --- END CLEAR STATE ---

        with st.spinner(f"Recreating '{LOG_COLLECTION}' to wipe data..."):
            try:
                # Recreate to ensure the collection exists but is empty
                client_reset.recreate_collection(
                    collection_name=LOG_COLLECTION,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
                st.success(f"✅ Successfully wiped all the datas in Qdrant collection")
                
            except Exception as e:
                st.error(f"Failed to wipe collection: {e}")

        st.rerun()

    st.markdown("---")
    
    # ----------------------------------------------------
    # 2. Reset Query Cache Collection (Minimal State Reset)
    # ----------------------------------------------------
    st.subheader("Reset Query Cache Collection")
    st.info(f"This clears all cached summaries and root-cause analyses stored")
    
    confirm_cache = st.checkbox(
        f"I understand — reset the Cache collection",
        key="cache_reset_confirmed",
        value=True  # Default state is ticked
    )

    if st.button("🟡 Reset Query Cache Collection Only", key="reset_cache_button") and confirm_cache:
        
        with st.spinner(f"Recreating collection to wipe cache..."):
            try:
                # Recreate to ensure the collection exists but is empty
                client_reset.recreate_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config={
                         "query_vec": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                    }
                )
                st.success(f"✅ Successfully wiped data in query cache collection")
                
            except Exception as e:
                st.error(f"Failed to wipe collection : {e}")

with tab3:
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)
    st.info("Clears all entries in the PostgreSQL logs table without affecting its structure, freeing up space and starting fresh")
    if st.button("Reset PostgreSQL Table"):
        conn = connect_pg()
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY")
            conn.commit()
        conn.close()

        # Reset logs DataFrame for chart/table
        logs_df_total = pd.DataFrame(columns=['severity', 'count'])
        styled_logs_df = logs_df_total.style  # empty styled DF

        st.success("All log records removed. PostgreSQL logs table is now empty ✅")
        st.rerun()

with tab4:
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)
    st.info("Enable template-only indexing with deduplication for faster processing while keeping essential details")
    fast_mode_ui = st.checkbox("⚡ Fast Indexing", value=True)

with tab5:
    st.markdown('<div style="padding-top:3rem;"></div>', unsafe_allow_html=True)
    st.info("Only the logs you select will be ingested into Qdrant (optional)")
    only_lvls = st.multiselect("Filter Levels", ["ERROR", "WARNING", "FATAL", "PANIC"])

def load_uploaded_file(uploaded_file):
    """Load .txt, .log files and return text content."""
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    text = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix in {".txt", ".log"}:
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        st.error("Unsupported file format. Use .txt/.log")
        return ""

    # Clean up uploaded tmp
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return text

def get_file_hash(file):
    """Generate a hash for the uploaded file content"""
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()

st.markdown("""
<style>
    /* Remove top padding from sidebar container */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar Heading -------------------
st.sidebar.markdown("""
<div style="margin-top:0px; text-align:center;">
    <h4 style='font-size:16px; margin:0; padding:0;'>🔎 Askpglog - PostgreSQL Log Analyzer</h4>
</div>
""", unsafe_allow_html=True)

# Add a vertical gap (you can adjust height in px)
st.sidebar.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ------------------- File uploader -------------------
# ------------------- File uploader -------------------
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}

uploaded_files = st.sidebar.file_uploader(
    "📁 Upload log files",
    type=["txt", "log"],
    accept_multiple_files=True,
    # CRITICAL: Use the dynamic key here
    key=st.session_state.file_uploader_key 
)
# -------------------------
# Qdrant upsert with retries/backoff
# -------------------------
def qdrant_upsert_points(points_batch):
    last_err = None
    for attempt in range(1, QDRANT_MAX_RETRIES + 1):
        try:
            client.upsert(collection_name=COLLECTION, points=points_batch, wait=False)
            return
        except Exception as e:
            last_err = e
            sleep_s = (QDRANT_BACKOFF_BASE ** attempt) + random.uniform(0, 0.5)
            sleep_s = min(sleep_s, 20.0)
            try: st.warning(f"Qdrant upsert retry {attempt}/{QDRANT_MAX_RETRIES} in ~{sleep_s:.1f}s ({type(e).__name__})")
            except Exception: pass
            time.sleep(sleep_s)
    raise RuntimeError(f"Qdrant upsert failed after {QDRANT_MAX_RETRIES} retries: {last_err}")

# -------------------------
# Indexing (FAST or FULL) + severity filter
# -------------------------
def upsert_events_qdrant(
    events,
    shard_size: int = SHARD_SIZE,
    fast_mode: bool = True,
    only_levels: set[str] | None = None
):
    if not events:
        return 0
    if only_levels:
        events = [e for e in events if (e.get("level") or "").upper() in only_levels]
        if not events:
            return 0

    total_upserts = 0
    first_dim = None

    if QDRANT_RESET and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    for s in range(0, len(events), shard_size):
        slice_events = events[s:s+shard_size]
        full_texts = [build_full_text(e) for e in slice_events]
        tpl_texts  = [fingerprint_text(t) for t in full_texts]

        if fast_mode:
            # deduplicate templates
            uniq_tpl = {}
            for idx, t in enumerate(tpl_texts):
                uniq_tpl.setdefault(t, []).append(idx)

            uniq_keys = list(uniq_tpl.keys())
            tpl_vecs_unique = embed_texts(uniq_keys)  # cached!

            dim = int(tpl_vecs_unique.shape[1])
            if first_dim is None:
                first_dim = dim
                recreate_base_collection(dim, fast=True)

            # map unique vecs back to all events
            vecs_tpl = [None] * len(slice_events)
            for k_idx, key in enumerate(uniq_keys):
                v = tpl_vecs_unique[k_idx]
                for event_idx in uniq_tpl[key]:
                    vecs_tpl[event_idx] = v

            points = []
            for e, v_tpl, t_full, t_tpl in zip(slice_events, vecs_tpl, full_texts, tpl_texts):
                pid = make_point_id(e)
                payload = {
                    "ts": int(e["ts"].timestamp()),
                    "ts_str": e["ts"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "tz": e.get("tz"),
                    "pid": e.get("pid"),
                    "ident": e.get("ident"),
                    "host_ip": e.get("host_ip"),
                    "port": e.get("port"),
                    "level": e.get("level"),
                    "message": e.get("message"),
                    "detail": e.get("detail"),
                    "statement": e.get("statement"),
                    "full_text": t_full,
                    "template_text": t_tpl,
                }
                vec = {"text_tpl": np.asarray(v_tpl, dtype="float32").tolist()}
                points.append(PointStruct(id=pid, vector=vec, payload=payload))

        else:
            # FULL mode: two vectors per point
            full_vecs = embed_texts(full_texts)
            tpl_vecs  = embed_texts(tpl_texts)

            dim = int(full_vecs.shape[1])
            if first_dim is None:
                first_dim = dim
                if not client.collection_exists(COLLECTION):
                    recreate_base_collection(dim, fast=False)

            points = []
            for e, v_full, v_tpl, t_full, t_tpl in zip(slice_events, full_vecs, tpl_vecs, full_texts, tpl_texts):
                pid = make_point_id(e)
                payload = {
                    "ts": int(e["ts"].timestamp()),
                    "ts_str": e["ts"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "tz": e.get("tz"),
                    "pid": e.get("pid"),
                    "ident": e.get("ident"),
                    "host_ip": e.get("host_ip"),
                    "port": e.get("port"),
                    "level": e.get("level"),
                    "message": e.get("message"),
                    "detail": e.get("detail"),
                    "statement": e.get("statement"),
                    "full_text": t_full,
                    "template_text": t_tpl,
                }
                vec = {
                    "text_full": np.asarray(v_full, dtype="float32").tolist(),
                    "text_tpl":  np.asarray(v_tpl,  dtype="float32").tolist()
                }
                points.append(PointStruct(id=pid, vector=vec, payload=payload))

        # batched upserts
        B = max(64, QDRANT_BATCH)
        for i in range(0, len(points), B):
            qdrant_upsert_points(points[i:i+B])

        total_upserts += len(points)
        try: st.info(f"Indexed shard {s//shard_size + 1} — {len(points)} events (total {total_upserts}).")
        except Exception: pass

    return total_upserts

# -------------------------
# Vector name detection (supports older/newer clients)
# -------------------------
def detect_vector_name_for_collection(collection: str) -> str | None:
    """
    Returns a named vector to use (e.g., 'text_full' or 'text_tpl').
    If the collection only has a single *unnamed* vector, returns None.
    Works across qdrant-client versions.
    """
    try:
        info = client.get_collection(collection)
        names: set[str] = set()

        # Newer: sometimes expose dict under .vectors_config
        vconf = getattr(info, "vectors_config", None)
        if isinstance(vconf, dict):
            names.update(vconf.keys())

        # Portable path (older clients): .config.params.vectors
        cfg = getattr(info, "config", None)
        params = getattr(cfg, "params", None) if cfg else None
        vecs = getattr(params, "vectors", None) if params else None

        # If vecs is a dict ⇒ named vectors; if it's VectorParams ⇒ single unnamed vector
        if isinstance(vecs, dict):
            names.update(vecs.keys())

        if "text_full" in names:
            return "text_full"
        if "text_tpl" in names:
            return "text_tpl"
        if names:
            return sorted(names)[0]

        # No named vectors detected ⇒ likely single unnamed vector
        return None
    except Exception:
        return None

# -------------------------
# Search / Context
# -------------------------
def qdrant_semantic_search(
    query: str,
    top_k: int = 50,
    level: str | None = None,
    t_start: int | None = None,
    t_end: int | None = None
):
    qv = embed_texts([query])[0].tolist()

    must = []
    if level:
        must.append(FieldCondition(key="level", match={"value": level}))
    if t_start or t_end:
        must.append(FieldCondition(key="ts", range=Range(gte=t_start, lte=t_end)))
    flt = Filter(must=must) if must else None

    vec_name = detect_vector_name_for_collection(COLLECTION)

    if vec_name is None:
        # Single unnamed vector → pass raw vector
        return client.search(
            collection_name=COLLECTION,
            query_vector=qv,
            query_filter=flt,
            with_payload=True,
            limit=top_k
        )

    # Named vector
    try:
        return client.search(
            collection_name=COLLECTION,
            query_vector=(vec_name, qv),
            query_filter=flt,
            with_payload=True,
            limit=top_k
        )
    except Exception:
        # Defensive fallback
        return client.search(
            collection_name=COLLECTION,
            query_vector=qv,
            query_filter=flt,
            with_payload=True,
            limit=top_k
        )

def expand_context_around_hit(hit, window_seconds=3600):
    ts = hit.payload["ts"]
    flt = Filter(must=[FieldCondition(key="ts", range=Range(gte=ts-window_seconds, lte=ts+window_seconds))])
    out, _ = client.scroll(
        collection_name=COLLECTION,
        limit=2000,
        with_payload=True,
        with_vectors=False,
        scroll_filter=flt
    )
    out = sorted(out, key=lambda p: (p.payload.get("ts", 0), p.payload.get("pid") or 0))
    return out

# Regex to find PIDs in DETAIL lines, e.g., "Process 25002 waits for ShareLock ... blocked by process 24891"
DETAIL_PID_RE = re.compile(r'Process (\d+) waits .*?blocked by process (\d+)', re.IGNORECASE)

def get_deadlock_hit_for_pid(hits, pid):
    """
    Return the first deadlock hit matching the given PID, 
    or any PIDs involved in the DETAIL lines of a deadlock.
    """
    for h in hits or []:
        pl = h.payload or {}
        msg = pl.get("message", "")
        hit_pid = pl.get("pid")

        if not msg or not DEADLOCK_LINE_RE.search(msg):
            continue

        # First, check if the hit PID matches directly
        if hit_pid == pid:
            return h

        # If not, check if the PID appears in the DETAIL lines of the deadlock
        detail_lines = pl.get("detail", "")
        for m in DETAIL_PID_RE.finditer(detail_lines):
            p1, p2 = int(m.group(1)), int(m.group(2))
            if pid in (p1, p2):
                return h

    return None

## -------------------------
# Deadlock helpers (REPLACEMENT)
# -------------------------
# Regex to detect "deadlock detected"
DEADLOCK_LINE_RE = re.compile(r'\bdeadlock detected\b', re.I)

# Regex to extract waiters/blockers from DETAIL
DEADLOCK_DETAIL_PAIR_RE = re.compile(
    r'Process\s+(?P<waiter>\d+)\s+waits.*?transaction\s+(?P<txid>\d+);.*?blocked by process\s+(?P<blocker>\d+)',
    re.I
)

def extract_deadlock_participants(payload_detail: str) -> dict:
    """Return dict with pairs of (waiter, blocker, txid) and unique PIDs."""
    pairs = []
    pids = set()
    if payload_detail:
        for m in DEADLOCK_DETAIL_PAIR_RE.finditer(payload_detail):
            w = int(m.group('waiter'))
            b = int(m.group('blocker'))
            txid = int(m.group('txid'))
            pairs.append((w, b, txid))
            pids.update([w, b])
    return {"pairs": pairs, "pids": pids}

def extract_pid_from_query(query: str):
    """
    Extract the first PID mentioned within square brackets [] in the query.
    Returns the PID as an integer, or None if not found.
    """
    m = re.search(r'\[(\d+)\]', query)
    if m:
        return int(m.group(1))
    return None

def fetch_pid_window(ts_center: int, pids: set[int], window_seconds: int = 120):
    """Fetch events around ts_center for given PIDs, sorted by timestamp."""
    if not pids:
        return []
    flt = Filter(must=[FieldCondition(key="ts", range=Range(
        gte=ts_center - window_seconds,
        lte=ts_center + window_seconds
    ))])
    out, next_page = client.scroll(COLLECTION, limit=2000, with_payload=True, with_vectors=False, scroll_filter=flt)
    all_pts = list(out)
    while next_page:
        out, next_page = client.scroll(COLLECTION, limit=2000, with_payload=True, with_vectors=False,
                                       scroll_filter=flt, offset=next_page)
        all_pts.extend(out)
    # Filter by PIDs
    all_pts = [p for p in all_pts if (p.payload or {}).get("pid") in pids]
    all_pts.sort(key=lambda p: (p.payload.get("ts", 0), p.payload.get("pid") or 0))
    return all_pts

def find_last_statement_before(ts_center: int, points_for_pid: list):
    """Return the last SQL statement at or before ts_center."""
    last_stmt, last_stmt_ts = None, None
    for p in points_for_pid:
        ts = p.payload.get("ts", 0)
        if ts > ts_center:
            break
        stmt = p.payload.get("statement")
        if stmt:
            last_stmt, last_stmt_ts = stmt, ts
        else:
            ft = (p.payload or {}).get("full_text", "")
            m = re.search(r'\bSTATEMENT:\s+(.*)$', ft, re.I | re.M)
            if m:
                last_stmt, last_stmt_ts = m.group(1).strip(), ts
    return last_stmt_ts, last_stmt

def render_deadlock_report(deadlock_hit, context_points):
    """Render deadlock report with full text, context, last statements, wait cycles, and fixes."""
    if not deadlock_hit:
        return

    pl = deadlock_hit.payload or {}
    ts = pl.get("ts")
    detail = pl.get("detail", "") or ""
    parts = extract_deadlock_participants(detail)
    pids, pairs = parts["pids"], parts["pairs"]

    st.markdown("## 🔒 Deadlock detected")
    st.code(pl.get("full_text", "")[:2000], language="text")
    if detail:
        st.write("**DETAIL:**")
        st.code(detail[:2000], language="text")

    # Context logs by PID
    by_pid = defaultdict(list)
    for p in context_points:
        by_pid[p.payload.get("pid")].append(p)

    st.markdown("### 🧭 Context (±120s) by PID")
    for pid, rows in by_pid.items():
        st.markdown(f"**PID {pid}**")
        for r in rows:
            st.text(r.payload.get("full_text", "")[:2000])

    # Last statements for each PID
    st.markdown("### 🧾 Involved SQL (last before deadlock)")
    for pid in sorted(by_pid.keys()):
        ts_stmt, stmt = find_last_statement_before(ts, by_pid[pid])
        if stmt:
            when = datetime.utcfromtimestamp(ts_stmt).strftime("%Y-%m-%d %H:%M:%S UTC") if ts_stmt else "(unknown)"
            st.markdown(f"**PID {pid}** — last statement @ {when}")
            st.code(stmt, language="sql")
        else:
            st.markdown(f"**PID {pid}** — *(no statement found in window; raise logging level or widen window)*")

    # Wait cycle
    if pairs:
        st.markdown("### 🔁 Wait cycle")
        for w, b, txid in pairs:
            st.write(f"- Process **{w}** waits on **txid {txid}**, blocked by **{b}**")

    # Suggested fixes
    st.markdown("### ✅ Targeted fixes")
    fixes = [
        "Ensure consistent lock ordering across code paths touching the same rows/tables.",
        "Keep transactions short; avoid `pg_sleep` within transactions.",
        "If FKs are involved, index all referencing columns to reduce lock scope.",
        "Add retry-on-deadlock with jitter to the affected statements only."
    ]
    for fix in fixes:
        st.write(f"- {fix}")

with app_tab:
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)

    # ------------------- Floating "Scroll to Top" Button -------------------
    st.markdown("""
    <div style="
        position: fixed; 
        top: 8.5rem;      
        right: 30px;      
        z-index: 9999;
    ">
        <a href="#top" 
        style="
            background-color:#cce5ff;
            color:black;
            font-weight:normal;
            text-decoration:none;
            padding:12px 18px;
            border-radius:12px;
            font-size:16px;
            box-shadow:0px 4px 8px rgba(0,0,0,0.2);
            cursor:pointer;
        ">
            ⬆️ Scroll to top
        </a>
    </div>
    """, unsafe_allow_html=True)

    # ------------------- Initialize session state -------------------
    for key in ["uploaded_data", "upload_sig", "first_upload_done", "index_ready"]:
        if key not in st.session_state:
            st.session_state[key] = {} if key=="uploaded_data" else None if key=="upload_sig" else False

    def compute_upload_signature(files):
        return tuple((f.name, getattr(f, "size", None)) for f in files) if files else None

    # ------------------- Ensure table exists at app start -------------------
    ensure_table_exists()

    # ------------------- File uploader UI -------------------
    if uploaded_files:
        st.sidebar.caption("Selected: " + ", ".join([f.name for f in uploaded_files]))
        for uf in uploaded_files:
            file_key = f"{uf.name}_{get_file_hash(uf)}"
            if file_key not in st.session_state.uploaded_data:
                st.session_state.uploaded_data[file_key] = {
                    "name": uf.name,
                    "content": load_uploaded_file(uf)
                }

            file_data = st.session_state.uploaded_data[file_key]
            st.subheader(f"📄 {file_data['name']} file")
            with st.expander("Preview file content", expanded=False):
                st.markdown(f"""
                <div style="
                    max-height:400px;
                    overflow:auto;
                    padding:10px;
                    border:1px solid #ddd;
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 14px;
                    user-select: text;
                ">{file_data['content'][:1000000]}</div>
                """, unsafe_allow_html=True)

    # ------------------- Detect file changes -------------------
    current_sig = compute_upload_signature(uploaded_files)
    files_changed = current_sig != st.session_state.upload_sig

    # ------------------- Ingest & Index controls -------------------
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        ingest_click = st.button("Ingest & Index")
    with col_btn2:
        auto_index_new = st.checkbox("Auto-index on change", value=True)

    # ------------------- Determine whether to index -------------------
    should_index_now = False
    first_upload = not st.session_state.get("first_upload_done", False)

    # Check if DB or Qdrant collection is empty
    db_empty = len(load_events_from_db()) == 0
    qdrant_empty = not qdrant_index_exists()

    if st.session_state.uploaded_data:
        if first_upload or ingest_click or (auto_index_new and files_changed) or db_empty or qdrant_empty:
            should_index_now = True
        elif ingest_click:
            should_index_now = True
        elif auto_index_new and files_changed:
            should_index_now = True

    # ------------------- Ingest -------------------
    all_events = []
    if should_index_now and uploaded_files:
        ensure_table_exists()
        for uf in uploaded_files:
            file_key = f"{uf.name}_{get_file_hash(uf)}"
            text = st.session_state.uploaded_data.get(file_key, {}).get("content", "")
            if not text:
                continue
            st.info(f"Parsing: {uf.name}")
            events = parse_events(text)
            inserted = insert_events_pg(events)
            st.success(f"Inserted {inserted} events from {uf.name}.")
            all_events.extend(events)

    # Create SQLAlchemy engine
    engine = create_engine("postgresql+psycopg2://postgres:accesspass@localhost:5433/tsdb")

    severity_query = """
    SELECT level AS severity, COUNT(*) AS count
    FROM uploaded_logs
    GROUP BY level
    ORDER BY
    CASE level
        WHEN 'PANIC' THEN 1
        WHEN 'FATAL' THEN 2
        WHEN 'ERROR' THEN 3
        WHEN 'WARNING' THEN 4
        WHEN 'NOTICE' THEN 5
        WHEN 'INFO' THEN 6
        WHEN 'LOG' THEN 7
        WHEN 'DEBUG' THEN 8
        ELSE 9
    END
    """
    try:
        logs_df = pd.read_sql(severity_query, engine)
    except Exception:
        logs_df = pd.DataFrame(columns=["severity", "count"])

    severity_order = ["PANIC","FATAL","ERROR","WARNING","NOTICE","INFO","LOG","DEBUG"]
    logs_df = logs_df.set_index('severity').reindex(severity_order, fill_value=0).reset_index()

    col1, col2 = st.columns([2,1])
    with col2:
        SHOW_ZERO_COUNTS = st.checkbox("Show severities with 0 count", value=True)

    display_df = logs_df[logs_df['count']>0] if not SHOW_ZERO_COUNTS else logs_df.copy()
    total_row = pd.DataFrame([{'severity':'TOTAL','count':display_df['count'].sum()}])
    logs_df_total = pd.concat([display_df,total_row],ignore_index=True)

    def highlight_total(row):
        return ['font-weight:bold']*len(row) if row['severity']=='TOTAL' else ['']*len(row)

    styled_logs_df = logs_df_total.style.apply(highlight_total, axis=1)

    # ------------------- Blue gradient for severity -------------------
    blue_colors = ["#3366cc","#4d79e6","#6699ff","#85adff","#a3c2ff","#c2d6ff","#e0ebff","#f2f9ff"]
    color_map = dict(zip(severity_order, blue_colors))

    with col1:
        fig = px.bar(
            display_df,
            x='severity',
            y='count',
            text='count',
            title='PostgreSQL Log Severity Counts',
            color='severity',
            color_discrete_map=color_map
        )
        fig.update_traces(
            textposition='outside',
            hovertemplate='Severity: %{x}<br>Count: %{y}',
            textfont_color='white'
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.dataframe(styled_logs_df.hide(axis="columns"), width="stretch")

    st.divider()

    # ------------------- Qdrant indexing -------------------
    if should_index_now and uploaded_files:
        with st.spinner("Indexing to Qdrant..."):
            count = upsert_events_qdrant(
                all_events if all_events else load_events_from_db(),
                shard_size=SHARD_SIZE,
                fast_mode=fast_mode_ui,
                only_levels=set(only_lvls) if only_lvls else None
            )
            st.success(f"Indexed/Upserted {count} events to Qdrant ({COLLECTION}).")

        st.session_state.upload_sig = current_sig
        st.session_state.index_ready = True
    else:
        if qdrant_index_exists():
            st.info("Using existing Qdrant index. Click **Ingest & Index** to re-index.")
            st.session_state.index_ready = True
        else:
            st.warning("No Qdrant index found yet. Upload logs and click **Ingest & Index**.")
            st.session_state.index_ready = False

    # -------------------------
    # Search panel
    # -------------------------
    q = st.text_input("Type a question or paste a log line")

    c1, c2, c3 = st.columns(3)
    with c1:
        level = st.selectbox("Filter by level", ["(any)", "LOG", "ERROR", "WARNING", "FATAL", "PANIC"])
        level_f = None if level == "(any)" else level
    with c2:
        start_str = st.text_input("Start time (optional, e.g. 2025-09-06 20:23)")
    with c3:
        end_str = st.text_input("End time (optional, e.g. 2025-09-06 20:25)")

    # Initialize session state flags

    if "last_hits" not in st.session_state:
        st.session_state.last_hits = None
    if "show_summary_button" not in st.session_state:
        st.session_state.show_summary_button = False
    if "expander_open" not in st.session_state:
        st.session_state.expander_open = True

    # -------------------------
    # Configuration
    # -------------------------
    QUERY_CACHE_COLLECTION = "query_cache"
    VECTOR_NAME = "query_vec"
    EMBEDDING_DIM = 1536  # your LLM embedding size

    # -------------------------
    # Connect to Qdrant
    # -------------------------
    # ⚠️ Use @st.cache_resource to ensure this initialization runs only ONCE.
    @st.cache_resource
    def init_qdrant_cache_client():
        """Initializes the Qdrant client and ensures the query_cache collection exists with correct config."""
        client = QdrantClient(url=f"http://localhost:6333")
        
        REQUIRED_VECTORS_CONFIG = {
            VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        }

        try:
            if client.collection_exists(QUERY_CACHE_COLLECTION):
                # Collection exists: Check configuration to only delete if necessary
                collection_info = client.get_collection(collection_name=QUERY_CACHE_COLLECTION)
                current_vectors = collection_info.config.params.vectors

                # Check if the expected vector name is present and matches the size
                is_correctly_configured = (
                    VECTOR_NAME in current_vectors and 
                    current_vectors[VECTOR_NAME].size == EMBEDDING_DIM
                )
                
                if not is_correctly_configured:
                    # Mismatch found (old 'default' config, wrong size, etc.). Force recreate.
                    print(f"⚠️ Vector config mismatch for '{QUERY_CACHE_COLLECTION}'. Deleting and recreating.")
                    client.delete_collection(collection_name=QUERY_CACHE_COLLECTION)
                    client.create_collection(
                        collection_name=QUERY_CACHE_COLLECTION,
                        vectors_config=REQUIRED_VECTORS_CONFIG
                    )
                    print(f"✅ Recreated collection '{QUERY_CACHE_COLLECTION}' with vector name '{VECTOR_NAME}'.")
                else:
                    # Configuration is correct. Do nothing.
                    print(f"✅ Collection '{QUERY_CACHE_COLLECTION}' is correctly configured.")
            else:
                # Collection does not exist. Create it for the first time.
                client.create_collection(
                    collection_name=QUERY_CACHE_COLLECTION,
                    vectors_config=REQUIRED_VECTORS_CONFIG
                )
                print(f"✅ Created collection '{QUERY_CACHE_COLLECTION}' with vector name '{VECTOR_NAME}'")

            return client

        except Exception as e:
            # Handle failure to connect/initialize
            st.error(f"Failed to initialize Qdrant query cache: {e}")
            # Raising the exception will stop the app from proceeding with a broken client
            raise 

    # -------------------------
    # Actual Qdrant Client Variable
    # This line runs once on startup and retrieves the cached client on subsequent runs.
    # -------------------------
    client = init_qdrant_cache_client()

    # -------------------------
    # Helpers
    # -------------------------
    def ensure_float_vector(vec):
        try:
            vec = np.asarray(vec, dtype=float).flatten()
            return vec.tolist()
        except Exception as e:
            print("❌ ensure_float_vector failed:", e, "input vec:", vec)
            raise ValueError(f"Embedding vector contains non-numeric values: {e}")

    def cache_query_answer(query: str, query_vec, answer: str):
        pid = str(uuid.uuid4())
        vec = ensure_float_vector(query_vec)
        payload = {
            "query": query,
            "answer": answer,
            "timestamp": int(time.time()),
            "usage_count": 1
        }
        client.upsert(
            collection_name=QUERY_CACHE_COLLECTION,
            points=[PointStruct(id=pid, vector={VECTOR_NAME: vec}, payload=payload)],
            wait=True
        )
        print(f"✅ Cached query '{query}' in Qdrant")

    def lookup_cached_answer(query_vec, threshold: float = 0.9):
        vec = ensure_float_vector(query_vec)
        results = client.search(
            collection_name=QUERY_CACHE_COLLECTION,
            query_vector=(VECTOR_NAME, vec),
            with_payload=True,
            limit=1
        )
        if results:
            hit = results[0]
            if hit.score >= threshold:
                return hit.payload.get("answer")
        return None

    def smart_cache_lookup(query_vec):
        cached = lookup_cached_answer(query_vec, threshold=0.95)
        if cached:
            return cached, "exact_semantic"
        cached = lookup_cached_answer(query_vec, threshold=0.85)
        if cached:
            return cached, "broad_semantic"
        return None, None

    # -------------------------
    # Main Search + Summary
    # -------------------------
    if st.button("Search"):
        if not q:
            st.warning("Enter a query.")
        elif not st.session_state.get("index_ready", False):
            st.error("No index available yet. Upload logs and click **Ingest & Index** first.")
        else:
            # Parse time filters
            try:
                t_start = int(dtparser.parse(start_str).timestamp()) if start_str.strip() else None
            except Exception:
                t_start = None
            try:
                t_end = int(dtparser.parse(end_str).timestamp()) if end_str.strip() else None
            except Exception:
                t_end = None

            with st.spinner("Searching events..."):
                hits = qdrant_semantic_search(q, top_k=50, level=level_f, t_start=t_start, t_end=t_end)

            if not hits:
                st.info("No results.")
            else:
                st.session_state.last_hits = hits

                with st.expander("📑 Top Matches and Context", expanded=True):
                    all_logs_for_rerank = []
                    hits_to_process = []

                    # -------------------------
                    # Deadlock detection
                    # -------------------------
                    if "deadlock detected" in q.lower():
                        query_pid = extract_pid_from_query(q)
                        deadlock_hit = get_deadlock_hit_for_pid(hits, query_pid)
                        st.session_state["deadlock_hit"] = deadlock_hit

                        if deadlock_hit:
                            pl = deadlock_hit.payload or {}
                            dl_ts = pl.get("ts")
                            detail = pl.get("detail", "") or ""
                            parts = extract_deadlock_participants(detail)
                            pids = parts["pids"]
                            pairs = parts["pairs"]

                            ctx = fetch_pid_window(dl_ts, pids, window_seconds=120)
                            render_deadlock_report(deadlock_hit, ctx)

                            last_statements = []
                            for pid in sorted(pids):
                                ts_stmt, stmt = find_last_statement_before(dl_ts, [p for p in ctx if p.payload.get("pid") == pid])
                                if stmt:
                                    when = datetime.utcfromtimestamp(ts_stmt).strftime("%Y-%m-%d %H:%M:%S UTC") if ts_stmt else "(unknown)"
                                    last_statements.append(f"PID {pid} — last statement @ {when}\n{stmt}")

                            wait_cycle_text = "\n".join([f"Process {w} waits on txid {txid}, blocked by {b}" for w, b, txid in pairs])
                            targeted_fixes = "\n".join([
                                "Ensure consistent lock ordering across code paths touching the same rows/tables.",
                                "Keep transactions short; avoid `pg_sleep` within transactions.",
                                "If FKs are involved, index all referencing columns to reduce lock scope.",
                                "Add retry-on-deadlock with jitter to the affected statements only."
                            ])
                            last_sql_text = "\n".join(last_statements)

                            st.session_state.reranked_logs = [
                                "\n\n".join([
                                    f"Deadlock message:\n{pl.get('full_text','')}",
                                    f"DETAIL:\n{detail}",
                                    f"Last SQL statements:\n{last_sql_text}",
                                    f"Wait cycle:\n{wait_cycle_text}",
                                    f"Targeted fixes:\n{targeted_fixes}"
                                ])
                            ]
                        else:
                            hits_to_process = hits[:5]
                    else:
                        hits_to_process = hits[:5]

                    # -------------------------
                    # Non-deadlock: display top hits & context
                    # -------------------------
                    if hits_to_process:
                        st.markdown("### 🎯 Top Matches")
                        window_sec = 1800
                        window_str = "±30 minutes"
                        all_context_logs = []
                        all_logs_for_rerank = []

                        for i, hit in enumerate(hits_to_process, start=1):
                            st.markdown(f"#### #{i}")
                            hit_text = hit.payload.get("full_text", "")
                            st.code(hit_text, language="text")
                            all_context_logs.append({
                                "ts": hit.payload.get("ts"),
                                "message": hit.payload.get("message"),
                                "full_text": hit_text
                            })

                            ctx = expand_context_around_hit(hit, window_seconds=window_sec)
                            for p in ctx:
                                ts = p.payload.get("ts")
                                msg = p.payload.get("message")
                                log_text = p.payload.get("full_text", "")
                                all_context_logs.append({"ts": ts, "message": msg, "full_text": log_text})

                        # Deduplicate logs
                        unique_context = []
                        seen_keys = set()
                        for log in all_context_logs:
                            key = (log["ts"], log["message"])
                            if key not in seen_keys:
                                unique_context.append(log)
                                seen_keys.add(key)

                        st.markdown(f"**Context ({window_str} + top hits)**")
                        for log in unique_context:
                            st.text(log["full_text"][:2000])
                            all_logs_for_rerank.append(log["full_text"])

                        # -------------------------
                        # Rerank logs for LLM
                        # -------------------------
                        if all_logs_for_rerank:
                            from sklearn.metrics.pairwise import cosine_similarity
                            import numpy as np

                            seen = set()
                            unique_logs = []
                            for log_text in all_logs_for_rerank:
                                if log_text not in seen:
                                    seen.add(log_text)
                                    unique_logs.append(log_text)

                            log_vecs = embed_texts(unique_logs)
                            q_vec = embed_texts([q])[0].reshape(1, -1)
                            sims = cosine_similarity(q_vec, log_vecs)[0]
                            top_idx = np.argsort(sims)[::-1][:20]
                            st.session_state.reranked_logs = [unique_logs[i] for i in top_idx]

                    st.session_state.expander_open = False

    # -------------------------
    # Summary Section & Button
    # -------------------------
    st.markdown('<div id="summary_section"></div>', unsafe_allow_html=True)

    if st.session_state.get("last_hits"):
        st.markdown("""
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999;">
            <a href="#summary_section" 
            onclick="window.parent.postMessage({{'show_summary': true}}, '*');"
            style="background-color:#cce5ff; color:black; text-decoration:none; padding:12px 18px; border-radius:12px; font-size:16px; box-shadow:0px 4px 8px rgba(0,0,0,0.2);">
                ✨ Go to Summary
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div id="top"></div>', unsafe_allow_html=True)

    if st.session_state.get("last_hits") and (not st.session_state.get("expander_open", False) or st.session_state.get("show_summary_button", False)):
        if st.button("✨ Generate Summary", key="summary_button"):
            if not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY not set.")
            else:
                with st.spinner("Checking cache..."):
                    try:
                        # -------------------------
                        # Embed the query
                        # -------------------------
                        q_vec = embed_texts([q])[0]
                        q_vec_list = np.asarray(q_vec, dtype=float).flatten().tolist()
                        print("🔹 Query embedding sample:", q_vec_list[:5])

                        # -------------------------
                        # Lookup in Qdrant cache
                        # -------------------------
                        cached_answer, cache_type = smart_cache_lookup(q_vec_list)
                        print("✅ Cache lookup result:", cached_answer, cache_type)

                        if cached_answer:
                            st.success(f"⚡ Served from cache ({cache_type})")
                            st.subheader("✅ Root Cause Summary")
                            st.markdown(cached_answer)
                        else:
                            # -------------------------
                            # Check if reranked logs are available
                            # -------------------------
                            logs_to_summarize = st.session_state.get("reranked_logs", [])
                            if not logs_to_summarize:
                                st.warning("No logs available for summarization. Perform a search first.")
                            else:
                                # Deduplicate logs
                                seen = set()
                                unique_logs = []
                                for log in logs_to_summarize:
                                    if log not in seen:
                                        seen.add(log)
                                        unique_logs.append(log)

                                summary_logs = [log[:2000] for log in unique_logs]
                                corpus = "\n\n---\n\n".join(summary_logs[:20])

                                # Display logs being sent
                                st.markdown(f"""
                                <textarea readonly style="width:100%; height:300px; background-color:white; color:black; border:1px solid #ddd; padding:10px; font-family:monospace; font-size:14px; overflow:auto; resize:none;">{corpus}</textarea>
                                """, unsafe_allow_html=True)

                                # -------------------------
                                # Call LLM
                                # -------------------------
                                system_prompt = (
                                    "You are a PostgreSQL SRE. "
                                    "Given PostgreSQL logs, produce a concise root-cause explanation and actionable steps."
                                )
                                user_prompt = (
                                    f"User Query:\n{q}\n\n"
                                    f"Relevant logs:\n{corpus}\n\n"
                                    "Instructions: Analyze the logs strictly in the context of the user's query. "
                                    "Provide a concise root-cause explanation, evidence with timestamps, "
                                    "and 3 actionable fixes directly addressing the query."
                                )

                                print("📝 Sending prompts to LLM...")
                                llm_response = call_openai_chat([
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ])
                                print("✅ LLM response received")

                                st.subheader("✅ Root Cause Summary")
                                st.markdown(llm_response)

                                # -------------------------
                                # Cache the answer in Qdrant
                                # -------------------------
                                cache_query_answer(q, q_vec_list, llm_response)
                                print("✅ Cached successfully")

                    except Exception as e:
                        import traceback
                        print("❌ Exception trace:", traceback.format_exc())
                        st.error(f"LLM summarization failed: {e}")
