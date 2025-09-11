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