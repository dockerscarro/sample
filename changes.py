```python
import streamlit as st

# Assuming logs is a list of log messages with their corresponding levels
logs = [
    {"level": "INFO", "message": "This is an info message."},
    {"level": "WARNING", "message": "This is a warning message."},
    {"level": "ERROR", "message": "This is an error message."},
    # Add more logs as needed
]

# Add a search box above the log display
search_query = st.text_input("Search logs:")

# Filter logs based on the search query
if search_query:
    filtered_logs = [
        log for log in logs if search_query.lower() in log["message"].lower()
    ]
else:
    filtered_logs = logs

# Display the filtered logs with color formatting
for log in filtered_logs:
    if log["level"] == "ERROR":
        st.markdown(f"<span style='color:red'>{log['message']}</span>", unsafe_allow_html=True)
    elif log["level"] == "WARNING":
        st.markdown(f"<span style='color:orange'>{log['message']}</span>", unsafe_allow_html=True)
    elif log["level"] == "INFO":
        st.markdown(f"<span style='color:green'>{log['message']}</span>", unsafe_allow_html=True)
    else:
        st.markdown(log['message'])
```
