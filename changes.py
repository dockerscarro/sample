```python
# Assuming 'logs' is a list of log messages and 'log_display' is the component displaying the logs

search_keyword = st.text_input("Search logs:")

# Filter logs based on the search keyword
if search_keyword:
    filtered_logs = [log for log in logs if search_keyword.lower() in log.lower()]
else:
    filtered_logs = logs

# Display filtered logs with existing color formatting
for log in filtered_logs:
    if "ERROR" in log:
        st.markdown(f"<span style='color:red'>{log}</span>", unsafe_allow_html=True)
    elif "WARNING" in log:
        st.markdown(f"<span style='color:orange'>{log}</span>", unsafe_allow_html=True)
    elif "INFO" in log:
        st.markdown(f"<span style='color:green'>{log}</span>", unsafe_allow_html=True)
    else:
        st.markdown(log)
```
