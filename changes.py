Here is the updated code to move the file upload to the left sidebar:

```python
### UPDATED START
st.set_page_config(page_title="Hybrid Log Search", layout="wide")
st.title("📚 Hybrid Log Search (Qdrant) ")

# Move file uploader to the sidebar
st.sidebar.markdown("Upload a PostgreSQL log file")
uploaded_file = st.sidebar.file_uploader("📁 Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
### UPDATED END

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
```

This code moves the file uploader to the left sidebar by using `st.sidebar.file_uploader` instead of `st.file_uploader`. The rest of the code remains the same.