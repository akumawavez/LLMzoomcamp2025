# ⚙️ Setup Guide

This guide will help you install and run the **Dutch Real Estate Buyers Assistant** on your local machine.

---

## 🧰 Prerequisites

Before running the project, ensure you have the following installed:

| Tool | Purpose | Download |
|------|----------|-----------|
| **Python 3.10+** | Core programming environment | [python.org/downloads](https://www.python.org/downloads/) |
| **Ollama** | Local LLM runtime | [ollama.ai](https://ollama.ai/) |
| **Qdrant** | Vector database for embeddings | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| **Streamlit** | Web interface | [streamlit.io](https://streamlit.io/) |
| **Docker** (optional) | Run Qdrant easily | [docker.com](https://www.docker.com/) |

---

## 🧩 Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/ajaimathew/dutch-real-estate-assistant.git
cd dutch-real-estate-assistant
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Qdrant (via Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Start Ollama
```bash
ollama serve
ollama pull llama3.2:3b
```

### 5. Run the Streamlit App
```python
streamlit run ChatApp_RAG_ollama_langchain.py
```

Then open the provided local URL (typically http://localhost:8501) in your browser.


⚙️ Environment Variables (Optional)

You can define environment variables to customize your setup:

Variable	Default	Description
OLLAMA_URL	http://localhost:11434	Ollama endpoint
QDRANT_URL	http://localhost:6333	Qdrant endpoint
CHAT_MODEL	llama3.2:3b	LLM used for responses
EMBEDDING_MODEL	nomic-embed-text	Embedding model used for document vectors

Create a .env file in the project root to override defaults.


✅ Verification

Once running, you should see two tabs in your browser app:

💬 Assistant – The chatbot interface.

🧠 Knowledge Base – Upload, list, and delete PDFs.

If these appear correctly, your setup is complete.

Authored by Ajai Mathew, 2025.