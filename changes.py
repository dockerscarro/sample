```python
### SECTION ###
uploaded_file = st.file_uploader("📁 Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])

# Function to color log messages based on their level
def color_log_message(level, message):
    if level == "ERROR":
        return f'<span style="color:red;">{message}</span>'
    elif level == "WARNING":
        return f'<span style="color:orange;">{message}</span>'
    elif level == "INFO":
        return f'<span style="color:green;">{message}</span>'
    else:
        return message  # Default to black for other levels

# Example usage in log display
for log in logs:
    st.markdown(color_log_message(log.level, log.message), unsafe_allow_html=True)
```
