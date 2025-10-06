# 💬 Usage Guide

This guide explains how to use the **Dutch Real Estate Buyers Assistant** once it’s running.

---

## 🧠 Overview

The app consists of two main tabs:

1. **💬 Assistant** – Interact with the chatbot.
2. **🧠 Knowledge Base** – Upload and manage property-related documents.

---

## 💬 Assistant Tab

This is your main chat interface.

### Steps:
1. In the input box, type your question (e.g., *“What are the tax implications of buying a second home in the Netherlands?”*).
2. Press **Enter** or click **Send**.
3. The chatbot responds using both:
   - Its built-in real estate knowledge.
   - Any uploaded document content relevant to your question.

### Notes:
- The assistant is domain-specific: Dutch real estate.
- If asked off-topic questions, it gently redirects users back on track.
- Context retrieval happens automatically from Qdrant.

---

## 🧠 Knowledge Base Tab

This tab manages your uploaded PDFs.

### Uploading a PDF
1. Click **Upload PDF** and select your file.
2. Wait for the progress spinner — the app will:
   - Extract text from each page.
   - Split it into 1000-character chunks.
   - Generate embeddings via Ollama.
   - Store vectors in Qdrant.

You’ll see a success message confirming the number of stored chunks.

### Viewing Stored Documents
All uploaded PDFs are listed under **Current Documents**.

### Deleting a Document
Click the 🗑️ **Delete** button next to a filename to remove it (including all associated embeddings).

---

## 🧩 Example Workflow

1. Upload a PDF about *Dutch mortgage options*.  
2. Ask:  
   > “What are the eligibility criteria for a mortgage as a foreigner?”  
3. The assistant will retrieve relevant text from your uploaded document and respond contextually.

---

## 💡 Tips
- Keep document filenames descriptive (e.g., `housing_taxes_2024.pdf`).
- The more detailed your documents, the richer the chatbot’s answers.
- You can upload multiple PDFs — they all go into the same **Qdrant collection**.

---

*Authored by Ajai Mathew, 2025.*
