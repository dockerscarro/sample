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
