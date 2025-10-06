# 🧱 System Architecture

This document explains the architecture and internal workflow of the **Dutch Real Estate Buyers Assistant**.

---

## 🧭 High-Level Overview


User → Streamlit UI → LangChain → Ollama (LLM & Embeddings) → Qdrant → PDF Knowledge Base

---

## 🧩 Components

### 1. Streamlit (Frontend)
- Provides the web interface with two tabs: Chat and Knowledge Base.
- Handles file uploads and user prompts.
- Displays conversation history and assistant responses.

### 2. Ollama (LLM Runtime)
- Runs locally (no API keys or internet required).
- Provides:
  - **Chat Model:** `llama3.2:3b` for generating responses.
  - **Embedding Model:** `nomic-embed-text` for vector representation of text chunks.

### 3. LangChain
- Orchestrates prompt creation, message handling, and RAG logic.
- Integrates Ollama and Qdrant seamlessly.
- Handles message memory and context injection.

### 4. Qdrant (Vector Database)
- Stores embeddings of PDF text chunks.
- Supports similarity search using cosine distance.
- Returns top-k relevant context sections for any user query.

### 5. PyPDF (Document Processing)
- Reads and extracts text from uploaded PDF files.
- Chunks text for embedding to maintain semantic integrity.

---

## 🔁 Workflow

1. **User Query:**  
   The user enters a question in the chat.

2. **Context Retrieval:**  
   The query is embedded and matched against stored vectors in Qdrant.

3. **Prompt Augmentation:**  
   The app constructs an enhanced prompt that includes relevant document excerpts.

4. **LLM Response:**  
   Ollama’s model (`llama3.2:3b`) generates a contextual, natural-language answer.

5. **Response Display:**  
   Streamlit renders the conversation dynamically.

---

## 🧠 Data Flow Diagram (Conceptual)

```text
PDF Upload
   ↓
Text Extraction → Chunking → Embedding → Qdrant Storage
                                               ↑
                                               |
Chat Input → Embedding → Similarity Search → Context Retrieval → LLM Response → Streamlit Display
```

## 🧰 Extensibility Ideas

Replace Ollama models with larger or domain-specific ones.

Use Docker Compose to containerize all components.

Add speech-to-text or multi-language support.