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
            PointStruct(id=ids[j], vector=embeddings[j].tolist(), payload={"text": texts[j]})
            for j in range(i, min(i + batch_size, len(texts)))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=batch_points)

    st.session_state.texts = texts
    st.session_state.embeddings = embeddings
    st.session_state.indexed = True
    st.success(f"✅ Indexed {len(texts)} logs into Qdrant.")

def build_text_loglevel_clusters_v2(
    source_collection="hybrid-search25", 
    profile_collection="text_loglevel_clusterss", 
    window=100
):
    """Fetch logs from Qdrant, group by template+level, build clusters, and insert into profile_collection."""
    import bisect
    from collections import defaultdict

    # Fetch all logs
    all_points = []
    points, next_page = client.scroll(
        collection_name=source_collection,
        with_payload=True,
        with_vectors=False,
        limit=1000
    )
    all_points.extend(points)
    while next_page is not None:
        points, next_page = client.scroll(
            collection_name=source_collection,
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=next_page
        )
        all_points.extend(points)

    # Unified regex for timestamp, pid, loglevel, and text
    time_pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+UTC\s+\[(\d+)]\s+([A-Z]+):\s+(.+)$"
    )
    
    messages = []
    for idx, p in enumerate(all_points):
        if not p.payload:
            continue
        txt = p.payload.get("text", "")
        match = time_pattern.match(txt)
        if not match:
            continue

        dt_str, pid_str, level, text_msg = match.groups()
        try:
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            ts = int(dt_obj.timestamp())
        except Exception:
            continue

        messages.append({
            "id": f"{p.id}_{ts}_{idx}",
            "timestamp": ts,
            "raw_timestamp": dt_str,
            "pid": int(pid_str),
            "text": text_msg.strip(),
            "log_level": level.upper()
        })

    # Sort messages chronologically
    messages_sorted = sorted(messages, key=lambda x: x["timestamp"])
    timestamps_sorted = [m["timestamp"] for m in messages_sorted]

    # Group by (template_text, log_level)
    text_loglevel_groups = defaultdict(list)
    for msg in messages_sorted:
        # Normalize template by stripping numbers (like IDs, query IDs, etc.)
        template_text = re.sub(r"\d+", "<NUM>", msg["text"]).strip()
        key = (template_text, msg["log_level"])
        text_loglevel_groups[key].append(msg)

    # Build clusters
    clusters = []
    for (template_text, log_level), msgs in text_loglevel_groups.items():
        cluster_messages = []
        seen_ids = set()
        for m in msgs:
            ts = m["timestamp"]
            if m["id"] not in seen_ids:
                cluster_messages.append(m)
                seen_ids.add(m["id"])
            start_idx = bisect.bisect_left(timestamps_sorted, ts - window)
            end_idx = bisect.bisect_right(timestamps_sorted, ts + window)
            for n in messages_sorted[start_idx:end_idx]:
                if n["id"] not in seen_ids:
                    cluster_messages.append(n)
                    seen_ids.add(n["id"])
        clusters.append({
            "cluster_id": hash((template_text, log_level)) % (10**12),
            "anchor_text": template_text,
            "log_level": log_level,
            "messages": cluster_messages,
            "profiling": {"cluster_size": len(cluster_messages)}
        })

    # Recreate profile collection
    if client.collection_exists(profile_collection):
        client.delete_collection(profile_collection)
    client.create_collection(
        collection_name=profile_collection,
        vectors_config=VectorParams(
            size=embedder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )

    # Insert clusters
    for cluster in clusters:
        cluster_text = " ".join([m["text"] for m in cluster["messages"]])
        vector = embedder.encode(cluster_text).tolist()
        client.upsert(
            collection_name=profile_collection,
            points=[PointStruct(id=cluster["cluster_id"], vector=vector, payload=cluster)]
        )

    # Save locally
    with open("text_loglevel_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)

    return len(clusters)

def extract_date_from_query(query):
    try:
        return parse(query, fuzzy=True).strftime('%Y-%m-%d')
    except Exception:
        return None

def qdrant_search(query, k=200, sim_threshold=0.15):
    target_date = extract_date_from_query(query)
    date_only_mode = False

    if target_date and any(
        phrase in query.lower()
        for phrase in ["from", "since", "after", "on", "logs from", "show logs from"]
    ) and not any(kw in query.lower() for kw in ["error", "fail", "division", "crash", "panic"]):
        date_only_mode = True

    if target_date and date_only_mode:
        conn = connect_pg()
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT message, pid, level, timestamp
                FROM {TABLE_NAME}
                WHERE DATE(timestamp) >= %s
                ORDER BY timestamp ASC
            """, (target_date,))
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return []

        texts = [
            f"{row[3].strftime('%Y-%m-%d %H:%M:%S.%f')} UTC [{row[1]}] {row[2]}: {row[0]}"
            for row in rows
        ]
    else:
        texts = st.session_state.get("texts", [])
        if not texts:
            return []

    query_vec = embedder.encode([query])[0]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    sim_scores = cosine_similarity([query_vec], embeddings).flatten()

    filtered = [(t, s) for t, s in zip(texts, sim_scores) if s >= sim_threshold]

    if not filtered:
        return []

    filtered.sort(key=lambda x: x[1], reverse=True)

    top_results = [t for t, _ in filtered[:k]]

    top_results.sort(key=extract_log_ts)

    return top_results

def extract_log_ts(log):
    try:
        ts_str = log.split(" UTC")[0].strip()
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None

def normalize_for_embedding(log):
    """
    Keep only severity + message, drop timestamp/PID.
    Example: "ERROR: could not connect to server: Connection refused"
    """
    try:
        return log.split("] ", 1)[1] 
    except Exception:
        return log

def cluster_logs(logs, n_clusters=10):
    if not logs:
        return []

    norm_texts = [normalize_for_embedding(l) for l in logs]
    embeddings = embedder.encode(norm_texts, show_progress_bar=False)

    k = min(n_clusters, len(logs))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    clusters = defaultdict(list)
    for label, log in zip(labels, logs):
        clusters[label].append(log)
    clustered_results = [sorted(group, key=extract_log_ts) for group in clusters.values()]

    return clustered_results

st.set_page_config(page_title="Hybrid Log Search", layout="wide")
st.title("📚 Hybrid Log Search (Qdrant) ")
st.markdown("Upload a PostgreSQL log file")

### UPDATED START c1b626ad ###
# placeholder
uploaded_file = st.file_uploader("📁 Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

# Function to color log messages based on their level
def color_log_message(level, message):
    if level == "ERROR":
        return f'<span style="color:red;">{message}</span>'
    elif level == "WARNING":
        return f'<span style="color:orange;">{message}</span>'
    elif level == "INFO":
        return f'<span style="color:green;">{message}</span>'
    else:
        return message  # Default to black for other levels

# Example usage in log display
for log in logs:
    st.markdown(color_log_message(log.level, log.message), unsafe_allow_html=True)
### UPDATED END c1b626ad ###

if uploaded_file:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("📥 Parsing and inserting logs into database..."):
            parsed_entries = load_and_parse_file(uploaded_file)
            ensure_table_exists()
            st.info("Table is ready")
            # Now insert logs
            inserted_count = insert_logs_pg(parsed_entries)
            
            if inserted_count > 0:
                st.success(f"✅ Inserted {inserted_count} new log entries into PostgreSQL.")
                st.session_state.indexed = False 
            else:
                st.info("ℹ️ No new entries inserted. Skipping indexing.")

            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_handled = True

    if not st.session_state.get("indexed", False):
        with st.spinner("⚙️ Indexing logs for search..."):
            all_logs = load_logs_from_db()
            index_texts(all_logs)
            cluster_status = st.empty()         
            build_text_loglevel_clusters_v2()         
            st.session_state.indexed = True
            st.success("✅ Indexing complete and ready for querying.")

else:
    st.info("📂 Please upload a file to insert and index logs.")

def query_cluster_profiles(user_query, top_k=10):
    """Semantic search on clusters in text_loglevel_clusterss collection."""
    q_vector = embedder.encode(user_query).tolist()  

    results = client.search(
        collection_name="text_loglevel_clusterss",
        query_vector=q_vector,
        limit=top_k,
        with_payload=True
    )

    if not results:
        return []

    clusters = [r.payload for r in results]
    return clusters

query = st.text_input("🔎 paste any log message:")
search_clicked = st.button("Search")

if search_clicked and query and st.session_state.get("indexed"):
    with st.spinner("🔍 Searching logs..."):

        dt_obj = None
        try:
            dt_obj = parser.parse(query, fuzzy=True)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            dt_obj = None

        clusters = query_cluster_profiles(query, top_k=5)
        all_cluster_logs = []

        for i, cluster in enumerate(clusters, 1):
            st.markdown(f"### 🧠 Cluster Result {i}")
            st.markdown(f"**Anchor Text:** {cluster.get('anchor_text', 'N/A')}")
            st.markdown(f"**Log Level:** {cluster.get('log_level', 'N/A')}")
            st.markdown(f"**Cluster Size:** {len(cluster.get('messages', []))} messages")

            # Collect logs from the cluster
            for msg in cluster.get("messages", []):
                ts = msg.get("raw_timestamp", msg.get("timestamp", "N/A"))
                pid = msg.get("pid", "N/A")  # if you stored PID in your payload
                level = msg.get("log_level", "UNKNOWN")
                text = msg.get("text", "")
                formatted_log = f"[{ts}] [{pid}] {level}: {text}"
                st.write(f"- {formatted_log}")
                all_cluster_logs.append(formatted_log)

            # Now fetch similar clusters
            cluster_text = " ".join([m.get("text", "") for m in cluster.get("messages", [])])
            cluster_vector = embedder.encode(cluster_text).tolist()
            similar_clusters = client.search(
                collection_name="text_loglevel_clusterss",
                query_vector=cluster_vector,
                limit=5,
                with_payload=True
            )

            st.markdown(f"#### 🔎 Top 4 Similar Clusters")
            for sim in similar_clusters:
                sim_payload = sim.payload
                if sim_payload.get("cluster_id") == cluster.get("cluster_id"):
                    continue
                st.markdown(f"- **Anchor Text:** {sim_payload.get('anchor_text', 'N/A')}")
                st.markdown(f"  **Log Level:** {sim_payload.get('log_level', 'N/A')}")
                for msg in sim_payload.get("messages", []):
                    ts = msg.get("raw_timestamp", msg.get("timestamp", "N/A"))
                    pid = msg.get("pid", "N/A")
                    level = msg.get("log_level", "UNKNOWN")
                    text = msg.get("text", "")
                    formatted_log = f"[{ts}] [{pid}] {level}: {text}"
                    st.write(f"    - {formatted_log}")
                    all_cluster_logs.append(formatted_log)

    all_logs_text = "\n".join(all_cluster_logs)
    structured_logs = [l.strip() for l in all_cluster_logs]
    all_logs_text_structured = "\n".join(f"{i+1}. {l}" for i, l in enumerate(structured_logs))

    if query:
        gpt_prompt = f"""
    You are a PostgreSQL Site Reliability Engineer (SRE).

    A user has made a query for a specific log message ("the Query Log").

    The Query Log is:
    "{query}"

    You are provided with the following **clustered log results**:
    {all_logs_text_structured}

    Instructions:

    1. STRICTLY use only the clusters provided above.
    - Do NOT invent or assume any logs.
    - Do NOT list all logs chronologically.
    - Focus on cluster summaries (Anchor Text, Log Level, Cluster Size).
    - Reference specific log messages only if they are critical for root cause.

    2. Identify the Query Log:
    - Find the single log that exactly matches the user's query.
    - Copy it exactly, including timestamp, PID, level, and message.

    3. Root Cause Analysis:
    - Identify which cluster(s) and log(s) directly caused the Query Log.
    - Copy causal logs exactly (timestamp, PID, level, message).
    - Do NOT include the Query Log itself as a causal log.
    - Warnings or past events may be referenced as **secondary contributing factors** only if they are clearly related.

    4. Explanation:
    - For each causal log, calculate the **time difference** from the Query Log precisely (e.g., "7 hours 7 minutes earlier").
    - Explain in PostgreSQL terms: table rewrites, connection drops, transaction conflicts, disk issues, etc.
    - Connect multiple causal logs into a **clear chain of events**.
    - Keep explanations concise (1–2 sentences per causal log).
    - Highlight why each causal log directly contributed to the Query Log.

    5. Possible Solution:
    - Suggest fixes based only on the provided clusters.
    - Include actionable PostgreSQL steps (adjust connection settings, monitor disk I/O, schedule maintenance, fix SQL issues).

    6. Output Format:
    - Query Log: <exact log>
    - Root Cause: <causal logs>
    - Explanation: <brief, PostgreSQL-aware reasoning with precise time differences>
    - Possible Solution: <concise recommended steps>
    """

        try:
            chat_model = ChatOpenAI(
                temperature=0,
                model="gpt-4o-mini",
                openai_api_key=OPENAI_API_KEY,
                max_tokens=1500
            )

            response = chat_model([
                HumanMessage(content="You are a PostgreSQL SRE. Analyze clustered logs efficiently, focusing on key clusters and causal logs with precise time-aware explanations."),
                HumanMessage(content=gpt_prompt)
            ])

            st.markdown("### 🧠 Log Analysis Summary & Root Cause")
            st.markdown(response.content.strip())

        except Exception as e:
            st.error(f"❌ GPT analysis failed: {e}")
