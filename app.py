import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import os

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Free AI Document Chatbot (RAG)")

# ----------------------------
# LOCAL EMBEDDING MODEL
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode([text])[0].tolist()

embeddings = LocalEmbeddings()

# ----------------------------
# SIMPLE CACHE FOR DB
# ----------------------------
@st.cache_resource
def build_db(texts):
    return FAISS.from_documents(texts, embeddings)

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

documents = []

if uploaded_file:

    if uploaded_file.name.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

    else:
        text = uploaded_file.read().decode("utf-8")
        documents = [Document(page_content=text)]

    # ----------------------------
    # CHUNKING
    # ----------------------------
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # ----------------------------
    # VECTOR DB
    # ----------------------------
    db = build_db(texts)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    st.success("✅ Document processed successfully!")

    # ----------------------------
    # LLM (FREE OPTION)
    # ----------------------------
    try:
        llm = Ollama(model="mistral")
        use_llm = True
    except:
        use_llm = False

    # ----------------------------
    # CHAT INPUT
    # ----------------------------
    query = st.text_input("Ask a question from your document:")

    if query:

        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        # ----------------------------
        # RESPONSE
        # ----------------------------
        if use_llm:
            response = llm.invoke(prompt)
        else:
            response = "⚠ Ollama not installed. Install it or use local logic."

        st.subheader("🤖 Answer")
        st.write(response)

        with st.expander("📌 Retrieved Context"):
            st.write(context)
