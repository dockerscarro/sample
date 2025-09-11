# Add a search box for log filtering
search_keyword = st.text_input("🔍 Filter logs by keyword:")

# Load all logs from the database
all_logs = load_logs_from_db()

# Filter logs based on the search keyword
if search_keyword:
    filtered_logs = [
        log for log in all_logs 
        if search_keyword.lower() in log.lower()
    ]
else:
    filtered_logs = all_logs

# Display the filtered logs
for log in filtered_logs:
    level = log.split("] [")[2].split(":")[0]  # Extract log level from the log string
    message = log.split(": ", 1)[1]  # Extract message from the log string
    st.markdown(color_log_message(level, message), unsafe_allow_html=True)
