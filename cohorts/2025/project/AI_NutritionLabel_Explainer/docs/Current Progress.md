I'll analyze this project to provide you with a comprehensive explanation of its architecture, features, and end-to-end functionality.

[5 tools called]

## AI Nutrition Label Explainer - Complete Project Overview

Based on my analysis, here's a comprehensive end-to-end explanation of this LLM Zoomcamp project:

---

## **Project Overview**

The **AI Nutrition Label Explainer** is a **RAG (Retrieval-Augmented Generation)** application that helps users understand nutrition labels and food products by answering questions using the Open Food Facts database. It's built as a Streamlit web application with local LLM capabilities using Ollama.

---

## **Architecture**

### **Tech Stack:**
- **Frontend**: Streamlit (web UI with chat interface)
- **LLM Framework**: LangChain
- **Language Model**: Ollama (local) - `llama3.2:3b`
- **Embeddings**: Ollama - `nomic-embed-text` (768-dimensional vectors)
- **Vector Database**: Qdrant (localhost:6333)
- **Data Source**: Open Food Facts dataset (CSV.GZ format)

### **System Components:**

```
┌─────────────────────────────────────────────────────────┐
│              Streamlit Web Interface                    │
│  (3 Tabs: Chat Assistant, Data Ingestion, Evaluation)  │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌─────────────────┐
│  Ollama LLM   │    │  Ollama Embed   │
│ (llama3.2:3b) │    │(nomic-embed-text)│
└───────────────┘    └────────┬─────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Qdrant Vector  │
                     │    Database     │
                     └─────────────────┘
                              ▲
                              │
                     ┌────────┴─────────┐
                     │ Data Ingestion   │
                     │ Pipeline         │
                     └──────────────────┘
                              ▲
                              │
                  ┌───────────┴────────────┐
                  │ Open Food Facts CSV.GZ │
                  └────────────────────────┘
```

---

## **Features**

### **1. Chat Assistant (💬 Tab)**
- **Interactive Q&A Interface**: Users can ask nutrition-related questions in natural language
- **Context-aware responses**: Uses retrieved product data to answer questions
- **Optional filters**: Category and brand filters to narrow down searches
- **Three retrieval modes**:
  - **Dense**: Pure vector similarity search
  - **Hybrid**: Vector search + metadata filtering (category/brand)
  - **Hybrid + Re-rank**: Hybrid with LLM-based reranking for better relevance

- **Performance metrics**: Displays response latency and number of retrieved documents
- **Retrieved context viewer**: Expandable section showing which products were used to generate the answer
- **Persistent chat history**: Maintains conversation context during the session

### **2. Data Ingestion (🍽 Tab)**
- **One-click ingestion**: Button to trigger the data pipeline
- **Automated data processing**:
  - Reads compressed Open Food Facts CSV (`.csv.gz`)
  - Filters products by 5 target categories:
    - Breakfast cereals
    - Beverages
    - Snacks
    - Dairies
    - Prepared meals
  - Cleans and formats text data
  - Generates embeddings using Ollama
  - Stores in Qdrant vector database

- **Performance optimizations**:
  - Streaming ingestion (1000 rows per chunk)
  - Batch upserts (25 vectors at a time)
  - Configurable record limit (5000 for demo purposes)

- **Real-time feedback**: Shows ingestion logs and error messages

### **3. Retrieval Evaluation (📊 Tab)**
- **A/B testing framework**: Compare three retrieval methods
- **Evaluation metrics**:
  - **HIT@K**: Percentage of queries where relevant results appear in top K
  - **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
  
- **Gold-standard test set**: Pre-defined queries with known relevant products
- **Visual comparison**: DataFrame and bar chart showing method performance
- **Configurable K parameter**: Slider to adjust top-K results (1-10)

---

## **End-to-End Workflow**

### **Phase 1: Data Ingestion** (`ingest_openfoodfacts.py`)

1. **Load Data**: Streams Open Food Facts CSV.GZ file (tab-separated)
2. **Filter**: 
   - Removes rows without product names
   - Keeps only products matching target categories
3. **Transform**:
   - Extracts key fields (product name, brand, categories, ingredients)
   - Formats nutrition facts (energy, fat, sugar, protein, salt, fiber)
   - Cleans text (removes line breaks, normalizes whitespace)
4. **Embed**:
   - Creates rich text descriptions combining all product info
   - Generates 768-dimensional vectors using `nomic-embed-text`
5. **Store**:
   - Creates Qdrant collection with cosine distance
   - Upserts points in batches with metadata
   - Stores text payload for retrieval

### **Phase 2: Query Processing** (Main App)

1. **User Input**: User types a nutrition question (e.g., "Is Corn Flakes high in sugar?")
2. **Embedding**: Query is embedded using the same `nomic-embed-text` model
3. **Retrieval**: Based on selected mode:
   - **Dense**: Pure vector similarity search (top-5)
   - **Hybrid**: Vector search filtered by category/brand metadata
   - **Hybrid + Re-rank**: Hybrid results reranked by LLM
4. **Context Building**: Retrieved documents combined into context text
5. **LLM Prompting**: 
   - Context + user question sent to Ollama LLM
   - Instructed to act as nutrition expert
   - Must cite product names when relevant
6. **Response**: Answer displayed with latency metrics and retrieved context

### **Phase 3: Evaluation** (Evaluation Tab)

1. **Test Queries**: Pre-defined gold-standard questions
2. **Multi-method Retrieval**: Runs all three methods on test set
3. **Relevance Checking**: Uses keyword matching to identify relevant results
4. **Metrics Calculation**: Computes HIT@K and MRR for each method
5. **Visualization**: Displays comparative performance

---

## **Key Technical Details**

### **Data Processing**
- **Format**: Tab-separated CSV with gzip compression
- **Fields used**: `product_name`, `brands`, `categories`, `ingredients_text`, `nutrition_grade_fr`, nutritional values per 100g
- **Text composition**: Concatenates all fields into semantic text for embedding

### **Vector Search**
- **Collection**: `openfoodfacts_filtered`
- **Vector size**: 768 dimensions
- **Distance metric**: Cosine similarity
- **Metadata fields**: Product name, brands, categories, URL, nutrition grade

### **Retrieval Strategies**
1. **Dense**: Embedding similarity only - fast but may retrieve irrelevant products
2. **Hybrid**: Adds metadata filters - better precision for specific queries
3. **Hybrid + Re-rank**: LLM judges relevance - highest quality but slower

### **Configuration**
All via environment variables with defaults:
- `OLLAMA_URL`: http://localhost:11434
- `QDRANT_URL`: http://localhost:6333
- `CHAT_MODEL`: llama3.2:3b
- `EMBED_MODEL`: nomic-embed-text

---

## **Unique Aspects**

1. **Fully local**: No API keys or cloud services required
2. **Educational project**: Demonstrates RAG concepts from LLM Zoomcamp
3. **Evaluation-driven**: Includes built-in retrieval method comparison
4. **Production-ready UI**: Streamlit chat interface with bottom-positioned input
5. **Flexible filtering**: Supports metadata-based search refinement
6. **Scalable design**: Streaming ingestion and batch processing for large datasets

This project demonstrates a complete RAG pipeline from data ingestion through retrieval evaluation, showcasing best practices in vector search, hybrid retrieval, and LLM-powered reranking.