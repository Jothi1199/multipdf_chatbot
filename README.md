# Multi-PDF Chatbot

An AI-powered chatbot that allows users to ask questions from multiple PDF documents using Retrieval-Augmented Generation (RAG).

---

## Features

- Supports multiple PDF documents
- Semantic search using embeddings
- Question Answering using Hugging Face models
- Retrieval-Augmented Generation (RAG)
- Interactive UI built with Streamlit
- Displays source context for answers

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB (Vector Database)
- Hugging Face Transformers
- Sentence Transformers

---

## Project Structure
multipdf_chatbot/
│
├── app.py
├── data/ # PDF files
├── requirements.txt
└── README.md

## Example Questions:

1. What are the skills mentioned?
2. Where is the person working?
3. What is the total experience?
4. What technologies are used?


## How It Works:

1. Loads multiple PDF documents
2. Splits text into chunks
3. Converts text into embeddings
4. Stores embeddings in ChromaDB
5. Retrieves relevant chunks based on query
6. Uses a QA model to extract answers

## Installation

```bash
git clone https://github.com/your-username/multipdf_chatbot.git
cd multipdf_chatbot
pip install -r requirements.txt

```

##  Run
streamlit run app.py