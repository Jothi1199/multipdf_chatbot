import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# UI TITLE
st.title("📄 Chat with Your Documents")

# LOAD DOCUMENTS
@st.cache_resource
def load_documents():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            docs.extend(loader.load())
    return docs

documents = load_documents()

# SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(documents)

# EMBEDDINGS
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# VECTOR DB
db = Chroma.from_documents(documents, embedding)

# QA MODEL
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
query = st.chat_input("Ask a question")

if query:
    # Show user message
    st.chat_message("user").write(query)

    # 🔥 Your existing RAG logic here
    results = db.max_marginal_relevance_search(query, k=5)

    best_answer = ""
    best_score = 0

    for doc in results:
        result = qa_pipeline(
            question=query,
            context=doc.page_content
        )

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    # Show assistant response
    st.chat_message("assistant").write(best_answer)

    # Save chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": best_answer})
