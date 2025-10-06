# 🏠 Dutch Real Estate Buyers Assistant (LangChain RAG)

**Author:** Ajai Mathew  
**Version:** 1.0  
**Status:** Active  
**License:** MIT  

An intelligent, privacy-preserving assistant that helps international buyers understand the **Dutch real estate market**, built using **LangChain**, **Ollama**, and **Qdrant**.

---

## 📚 Overview

The **Dutch Real Estate Buyers Assistant** is a Streamlit-based chatbot powered by **Retrieval-Augmented Generation (RAG)**.  
Users can upload property or mortgage-related PDFs, and the chatbot will answer questions using context retrieved from those documents.

It’s designed for:
- International buyers exploring Dutch property regulations.
- Demonstrating how to integrate **local LLMs** and **vector databases** for contextual Q&A.
- Showcasing **privacy-first AI** using locally hosted tools.

---

## ✨ Key Features
- 💬 **Conversational Interface:** Ask real estate questions in natural language.
- 📄 **Contextual Retrieval:** Uses your uploaded PDFs for informed answers.
- 🧠 **Vector Database:** Stores and retrieves embeddings via Qdrant.
- 🤖 **Local LLM Inference:** Runs Ollama models entirely on your machine.
- ⚙️ **Streamlit Interface:** Simple, clean, and interactive.
- 🔒 **Private and Secure:** No external API calls.

---

## 📁 Documentation
This project’s documentation is split into modular files for clarity:

| Section | Description |
|----------|--------------|
| [Setup Guide](docs/setup.md) | How to install and run the app locally |
| [Usage Guide](docs/usage.md) | How to upload, chat, and manage PDFs |
| [Architecture](docs/architecture.md) | How RAG, Ollama, and Qdrant work together |
| [Evaluation Criteria](docs/evaluation.md) | Rubric and project assessment notes |
| [Contributing](docs/contributing.md) | Guidelines for collaboration and updates |

---

## 🖼️ Visuals

**App Screenshot:**
![Chat Interface](assets/app_screenshot.png)

**System Architecture:**
![Architecture Diagram](assets/architecture_diagram.png)

*(Replace the above with your own images or generated diagrams.)*

---

## 🚀 Quick Start


# Clone repository
```bash
git clone https://github.com/ajaimathew/dutch-real-estate-assistant.git
cd dutch-real-estate-assistant
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

# Start Ollama
```bash
ollama serve
ollama pull llama3.2:3b
```

# Run Streamlit app
```python
streamlit run ChatApp_RAG_ollama_langchain.py
```
