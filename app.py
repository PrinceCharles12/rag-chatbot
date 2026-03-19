import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

st.title("📄 AI Document Chatbot (RAG)")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ API key not found. Add it in Streamlit Secrets.")
else:
    # Upload file (PDF + TXT)
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])

    if uploaded_file:
        file_name = uploaded_file.name

        # -------- PDF --------
        if file_name.endswith(".pdf"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

        # -------- TXT --------
        elif file_name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            documents = [Document(page_content=text)]

        # -------- Processing --------
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(texts, embeddings)

        llm = ChatOpenAI(openai_api_key=api_key)
        retriever = db.as_retriever()

        # User query
        query = st.text_input("Ask a question from your document:")

        if query:
            docs = retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])

            prompt = f"""
Answer the question based on the context below:

Context:
{context}

Question:
{query}
"""

            response = llm.predict(prompt)

            st.subheader("🤖 Answer:")
            st.write(response)
