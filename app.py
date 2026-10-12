import streamlit as st
from rag_pipeline import (
    extract_text, chunk_text,
    search_chunks, generate_answer
)

st.set_page_config(
    page_title="DocuChat",
    page_icon="📄"
)

st.title("📄 DocuChat")
st.subheader("Upload a PDF and ask questions!")

api_key = st.sidebar.text_input(
    "Enter Gemini API Key", type="password"
)

uploaded_file = st.file_uploader(
    "Upload your PDF", type="pdf"
)

if uploaded_file and api_key:
    with st.spinner("Processing your document..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        st.success(f"✅ Document processed! "
                  f"({len(chunks)} chunks created)")

    st.subheader("💬 Ask questions about your document")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Finding answer..."):
            relevant_chunks = search_chunks(query, chunks)
            answer = generate_answer(
                query, relevant_chunks, api_key
            )

        st.subheader("✅ Answer:")
        st.write(answer)

        with st.expander("📚 Source chunks used:"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.divider()
else:
    st.info("👈 Enter your Gemini API key "
           "and upload a PDF to get started!")