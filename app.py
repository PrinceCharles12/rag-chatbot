import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 Free AI Document Chatbot (RAG)")

# ---------------- MODEL ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- EMBEDDING WRAPPER ----------------
class EmbeddingsWrapper:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode([text])[0].tolist()

embeddings = EmbeddingsWrapper()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:

    # ---------------- PDF ----------------
    if uploaded_file.name.endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

    # ---------------- TXT ----------------
    else:
        text = uploaded_file.read().decode("utf-8")
        documents = [Document(page_content=text)]

    # ---------------- CHUNKING ----------------
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # ---------------- VECTOR DB ----------------
    db = FAISS.from_documents(texts, embeddings)

    st.success("✅ File processed successfully!")

    # ---------------- QUERY ----------------
    query = st.text_input("Ask a question from your document:")

    if query:

        # SEARCH
        docs = db.similarity_search(query, k=3)

        # CONTEXT
        context = "\n".join([d.page_content for d in docs])

        # SIMPLE ANSWER GENERATION
        answer = f"""
📌 Answer based on your document:

{context}

---

❓ Question: {query}
"""

        st.subheader("🤖 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            st.write(context)
