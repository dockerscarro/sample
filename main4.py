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
