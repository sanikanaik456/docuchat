import streamlit as st

from rag_pipeline import (
    extract_text,
    chunk_text,
    search_chunks_tfidf,
    generate_answer,
)

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")

st.title("📄 DocuChat")
st.markdown("*Upload a PDF and ask questions about it.*")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.caption("Retrieval: TF-IDF over 250-word chunks")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Reading the PDF..."):
            text = extract_text(uploaded_file)
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["chunks"] = chunk_text(text, chunk_size=250) if text.strip() else []

    chunks = st.session_state["chunks"]

    if not chunks:
        st.warning("No readable text found. The PDF may be a scanned image.")
    else:
        st.success(f"Ready: {len(chunks)} chunks from {uploaded_file.name}")
        query = st.text_input("Ask a question about the document")

        if st.button("Get answer", type="primary"):
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar.")
            elif not query.strip():
                st.error("Please type a question.")
            else:
                with st.spinner("Searching and generating an answer..."):
                    try:
                        relevant = search_chunks_tfidf(query, chunks)
                        answer = generate_answer(query, relevant, api_key)
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
                    else:
                        st.subheader("Answer")
                        st.write(answer)
                        with st.expander("Passages sent to Gemini"):
                            for i, chunk in enumerate(relevant, 1):
                                st.markdown(f"Passage {i}")
                                st.write(chunk[:1500])