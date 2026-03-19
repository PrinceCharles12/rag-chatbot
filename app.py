import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

st.title("📄 AI Document Chatbot (RAG)")

# Get API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ API key not found. Add it in Streamlit Secrets.")
else:
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])

    if uploaded_file:
        # PDF case
        if uploaded_file.type == "application/pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
                
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

    # TXT case
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

        from langchain.schema import Document
        documents = [Document(page_content=text)]

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split text
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Store in FAISS
        db = FAISS.from_documents(texts, embeddings)

        # LLM
        llm = ChatOpenAI(openai_api_key=api_key)

        # RAG Chain
        retriever = db.as_retriever()
        # Input
        query = st.text_input("Ask a question from your document:")
        if query:
            # Get relevant docs
            docs = retriever.get_relevant_documents(query)
            # Create context
            context = " ".join([doc.page_content for doc in docs])
    
            # Create prompt
            prompt = f"""Answer the question based on the context below:

        Context:
        {context}

        Question:
        {query}
        """

            # Get response
            response = llm.predict(prompt)
    
            st.write("🤖 Answer:", response)
