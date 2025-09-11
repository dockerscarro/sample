
import streamlit as st

def main():
    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_details = {"FileName":file.name,"FileType":file.type,"FileSize":file.size}
            st.sidebar.write(file_details)

if __name__ == '__main__':
    main()
