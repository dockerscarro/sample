st.set_page_config(page_title="Hybrid Log Search", layout="wide")
st.title("📚 Hybrid Log Search (Qdrant) ")
st.markdown("Upload a PostgreSQL log file")

uploaded_file = st.file_uploader("📁 Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

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
