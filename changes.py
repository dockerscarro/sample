```python
### SECTION 504a7615 ###
def color_log(level, message):
    if level == "ERROR":
        return f"<span style='color:red;'>{message}</span>"
    elif level == "WARNING":
        return f"<span style='color:orange;'>{message}</span>"
    elif level == "INFO":
        return f"<span style='color:green;'>{message}</span>"
    else:
        return f"<span style='color:black;'>{message}</span>"

### SECTION 0ce14625 ###
for msg in cluster.get("messages", []):
    ts = msg.get("raw_timestamp", msg.get("timestamp", "N/A"))
    pid = msg.get("pid", "N/A")  # if you stored PID in your payload
    level = msg.get("log_level", "UNKNOWN")
    text = msg.get("text", "")
    formatted_log = f"[{ts}] [{pid}] {level}: {text}"
    st.markdown(color_log(level, formatted_log), unsafe_allow_html=True)
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
        st.markdown(color_log(level, formatted_log), unsafe_allow_html=True)
        all_cluster_logs.append(formatted_log)
```
