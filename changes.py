```python
# placeholder
search_keyword = st.text_input("Search logs", "").strip().lower()

filtered_logs = [
    log for log in logs 
    if search_keyword in log.lower() or not search_keyword
]

for log in filtered_logs:
    if "ERROR" in log:
        st.markdown(f"<span style='color:red'>{log}</span>", unsafe_allow_html=True)
    elif "WARNING" in log:
        st.markdown(f"<span style='color:orange'>{log}</span>", unsafe_allow_html=True)
    elif "INFO" in log:
        st.markdown(f"<span style='color:green'>{log}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:black'>{log}</span>", unsafe_allow_html=True)
```
