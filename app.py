import streamlit as st
import os
from langchain_openai import ChatOpenAI

st.title("📄 AI Document Chatbot")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key not found. Please add it in Streamlit Secrets.")
else:
    llm = ChatOpenAI(openai_api_key=api_key)

    query = st.text_input("Ask something:")

    if query:
        response = llm.predict(query)
        st.write("🤖 Answer:", response)
