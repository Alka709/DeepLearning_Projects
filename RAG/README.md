# RAG Pipeline using LangChain, FAISS, and Ollama

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using:

- LangChain
- FAISS Vector Store
- HuggingFace Embeddings
- Ollama (Llama 3)
- PyPDFLoader

The system loads a PDF document, splits it into chunks, generates embeddings, stores them in FAISS, and retrieves relevant context to answer user queries using a local LLM.

---

# Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- Ollama
- Llama3

---

# Project Workflow

```text
PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embeddings
 ↓
FAISS Vector Store
 ↓
Retriever
 ↓
Ollama Llama3
 ↓
Generated Answer
```

---

# Installation

## Clone Repository

```bash
git clone <your-repo-url>
```

## Create Virtual Environment

```bash
python -m venv myvenv
```

## Activate Virtual Environment

### Windows

```bash
myvenv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install Ollama:

https://ollama.com/download

Pull the Llama3 model:

```bash
ollama pull llama3
```

---

# Run Project

```bash
python RAG.py
```

---

# Sample Query

```python
"Give me the gist of RAG in 3 sentences"
```

---

# Sample Output

```text
RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to improve response quality. Relevant information is first retrieved from external documents and then provided to the language model as context. This helps generate more accurate and context-aware answers.
```

---

# Features

- Local RAG pipeline
- Free embeddings using HuggingFace
- Local LLM inference using Ollama
- Semantic search using FAISS
- PDF-based question answering

---
