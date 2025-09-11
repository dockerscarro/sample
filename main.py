```python
import streamlit as st
import os
import tempfile
import uuid
import re
import json
import bisect
import numpy as np
import psycopg2
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from datetime import datetime, timezone
from dateutil import parser
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

PG_HOST = os.environ["PG_HOST"]
PG_DB = os.environ["PG_DB"]
PG_USER = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]
PG_PORT = int(os.environ.get("PG_PORT", "5432"))  # only port has a safe default
TABLE_NAME = os.environ.get("TABLE_NAME", "uploaded_logs")

EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

COLLECTION_NAME = "hybrid-search25"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")   # "qdrant" = service name in docker-compose
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

client = QdrantClient(url=qdrant_url)

import time
import requests

def wait_for_qdrant(host, port, timeout=30):
    url = f"http://{host}:{port}/collections"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Qdrant not ready after waiting.")

# Call this before any collection operations
wait_for_qdrant(QDRANT_HOST, QDRANT_PORT)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment.")

LOG_PATTERN = re.compile(r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) UTC \[(?P<pid>\d+)] (?P<level>[A-Z]+):\s+(?P<message>.+)')

def connect_pg():
    return psycopg2.connect(
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT
    )

def ensure_table_exists():
    conn = connect_pg()
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                message TEXT NOT NULL,
                pid INTEGER NOT NULL,
                level TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL
            )
        """)
        conn.commit()
    conn.close()


def load_and_parse_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".txt":
        loader = TextLoader(tmp_path)
    elif suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error("Unsupported file format.")
        return []

    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    return parse_logs_from_text(text)

def parse_logs_from_text(text):
    entries = []
    for line in text.splitlines():
        match = LOG_PATTERN.match(line.strip())
        if match:
            ts = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S.%f")
            entries.append((
                match.group("message").strip(),
                int(match.group("pid")),
                match.group("level"),
                ts
            ))
    return entries

def insert_logs_pg(rows):
    if not rows:
        return 0

    conn = connect_pg()
    with conn.cursor() as cur:
        inserted_count = 0
        for row in rows:
            cur.execute(f"""
                SELECT 1 FROM {TABLE_NAME}
                WHERE message = %s AND pid = %s AND level = %s AND timestamp = %s
                LIMIT 1
            """, row)
            if not cur.fetchone():
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} (message, pid, level, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, row)
                inserted_count += 1

        conn.commit()
    conn.close()
    return inserted_count

def load_logs_from_db():
    conn = connect_pg()
    with conn.cursor() as cur:
        cur.execute(f"SELECT message, pid, level, timestamp FROM {TABLE_NAME}")
        rows = cur.fetchall()
    conn.close()
    return [
        f"{row[3].strftime('%Y-%m-%d %H:%M:%S.%f')} UTC [{row[1]}] {row[2]}: {row[0]}"
        for row in rows
    ]

def setup_qdrant(dimension):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

def index_texts(texts, batch_size=500):
    if not texts:
        st.warning("No logs available to index.")
        return

    embeddings = embedder.encode(texts, show_progress_bar=True)

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        st.error(f"Unexpected embedding shape: {getattr(embeddings, 'shape', 'None')}")
        return

    ids = [str(uuid.uuid4()) for _ in texts]

    # Create or reset collection
    VECTOR_SIZE = embeddings.shape[1]
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

    # Batch upserts
    for i in range(0, len(texts), batch_size):
        batch_points = [
            PointStruct(id=ids[j], vector=embeddings[j].tolist(),