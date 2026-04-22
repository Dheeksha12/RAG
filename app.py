import streamlit as st
import tempfile
from rag_pipeline import create_vector_store, create_qa_chain

st.set_page_config(page_title="RAG Document QA", layout="wide")

st.title("📄 Document Question Answering System (RAG)")

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF Uploaded Successfully!")

    with st.spinner("Processing document..."):
        vectorstore = create_vector_store(pdf_path)
        qa_chain = create_qa_chain(vectorstore)

    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain(question)

        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Retrieved Source Chunks")

        for doc in result["source_documents"]:
            st.write(doc.page_content)
            st.write("---")