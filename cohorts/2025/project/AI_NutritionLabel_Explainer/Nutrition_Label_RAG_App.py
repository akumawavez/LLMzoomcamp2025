"""
AI Nutrition Label Explainer – Streamlit RAG App
------------------------------------------------
Tabs:
- 💬 Ask Nutrition Assistant: Chat interface powered by Ollama + Qdrant.
- 🍽 Data Ingestion: Runs ingestion of Open Food Facts data (.csv.gz).
"""
"""
AI Nutrition Label Explainer – LangChain + Streamlit version
Local, no FastAPI. Includes evaluation of retrieval methods.
"""

import os
import time
import subprocess
import pandas as pd
import streamlit as st
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from sklearn.metrics import label_ranking_average_precision_score

# =============================================================================
# --- Configuration ---
# =============================================================================
st.set_page_config(page_title="AI Nutrition Label Explainer", page_icon="🥗", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "openfoodfacts_filtered"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# LangChain models
llm = ChatOllama(model=CHAT_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL)

# =============================================================================
# --- Utility: LangChain Retriever Wrappers ---
# =============================================================================

def make_dense_retriever() -> BaseRetriever:
    """Plain dense retrieval using embeddings only."""
    vectordb = Qdrant(client=qdrant_client, collection_name=COLLECTION, embeddings=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 5})

def make_hybrid_retriever(category: str | None = None, brand: str | None = None) -> BaseRetriever:
    """Dense retrieval constrained by metadata (hybrid)."""
    vectordb = Qdrant(client=qdrant_client, collection_name=COLLECTION, embeddings=embeddings)
    filter_conds = []
    if category:
        filter_conds.append(FieldCondition(key="categories", match=MatchText(text=category)))
    if brand:
        filter_conds.append(FieldCondition(key="brands", match=MatchText(text=brand)))
    if filter_conds:
        flt = Filter(must=filter_conds)
        return vectordb.as_retriever(search_kwargs={"k": 5, "filter": flt})
    return vectordb.as_retriever(search_kwargs={"k": 5})

def rerank_with_llm(query: str, docs: List[Document]) -> List[Document]:
    """Optional lightweight LLM reranker."""
    if not docs:
        return []
    text_block = "\n".join([f"[{i}] {d.page_content[:200]}" for i, d in enumerate(docs, 1)])
    prompt = f"""Re-rank the following documents for the query:
Query: {query}
Docs:
{text_block}
Output a comma-separated list of document indices (best to worst).
"""
    resp = llm.invoke(prompt).content
    order = [int(x.strip()) for x in resp.split(",") if x.strip().isdigit()]
    ranked = [docs[i-1] for i in order if 0 < i <= len(docs)]
    return ranked + [d for d in docs if d not in ranked]

# =============================================================================
# --- Tabs ---
# =============================================================================
tab_chat, tab_ingest, tab_eval = st.tabs(["💬 Chat Assistant", "🍽 Data Ingestion", "📊 Retrieval Evaluation"])

# -------------------------------------------------------------------------
# 💬 CHAT TAB
# -------------------------------------------------------------------------
with tab_chat:
    st.title("🥗 AI Nutrition Label Explainer (LangChain + Qdrant)")

    category = st.text_input("Category filter (optional)")
    brand = st.text_input("Brand filter (optional)")
    mode = st.selectbox("Retrieval Mode", ["Dense", "Hybrid", "Hybrid + Re-rank"])

    if "history" not in st.session_state:
        st.session_state.history = []

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    query = st.chat_input("Ask a question about a product or nutrient...")
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Build retriever according to mode
        retriever = make_dense_retriever() if mode == "Dense" else make_hybrid_retriever(category, brand)
        docs = retriever.get_relevant_documents(query)
        if mode == "Hybrid + Re-rank":
            docs = rerank_with_llm(query, docs)

        context_text = "\n\n".join([d.page_content for d in docs])
        prompt = f"""You are a nutrition expert. 
Answer the user's question using ONLY the following context (Open Food Facts product data). 
Cite product names if relevant.

Context:
{context_text}

Question: {query}
"""
        t0 = time.time()
        answer = llm.invoke(prompt).content
        latency = round((time.time() - t0) * 1000, 1)

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Latency: {latency} ms, Retrieved docs: {len(docs)}")
        st.session_state.history.append({"role": "assistant", "content": answer})

        with st.expander("🔍 Retrieved Context"):
            for d in docs:
                st.markdown(f"- **{d.metadata.get('product_name','Unknown')}** — {d.metadata.get('categories','')}")
                st.caption(d.page_content[:300] + "...")

# -------------------------------------------------------------------------
# 🍽 INGESTION TAB
# -------------------------------------------------------------------------
with tab_ingest:
    st.header("🍽 Data Ingestion (Open Food Facts)")
    script_path = os.path.join(os.path.dirname(__file__), "ingest_openfoodfacts.py")
    if not os.path.exists(script_path):
        st.error("Ingestion script not found!")
    else:
        if st.button("🚀 Run Ingestion"):
            with st.spinner("Running ingestion..."):
                result = subprocess.run(["python", script_path], capture_output=True, text=True)
                st.code(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
                else:
                    st.success("Ingestion completed successfully!")

# -------------------------------------------------------------------------
# 📊 EVALUATION TAB
# -------------------------------------------------------------------------
with tab_eval:
    st.header("📊 Retrieval Evaluation")
    st.write("""
    Evaluate Dense vs Hybrid vs Hybrid + Re-rank retrieval on a small gold-standard set.
    Metrics: HIT@K, MRR, Recall.
    """)

    gold_examples = [
        {"query": "Is Corn Flakes high in sugar?", "relevant_keyword": "Corn Flakes"},
        {"query": "Which yogurt has less fat?", "relevant_keyword": "Yogurt"},
        {"query": "Is this drink low in sodium?", "relevant_keyword": "Beverage"},
    ]

    k = st.slider("Top K", 1, 10, 5)
    methods = ["Dense", "Hybrid", "Hybrid + Re-rank"]
    results = []

    if st.button("Run Evaluation"):
        for method in methods:
            hits, rr, total = 0, 0.0, 0
            for g in gold_examples:
                query, keyword = g["query"], g["relevant_keyword"]
                if method == "Dense":
                    retriever = make_dense_retriever()
                else:
                    retriever = make_hybrid_retriever()
                docs = retriever.get_relevant_documents(query)
                if method == "Hybrid + Re-rank":
                    docs = rerank_with_llm(query, docs)
                retrieved_titles = [d.metadata.get("product_name","") for d in docs[:k]]
                found_ranks = [i+1 for i, t in enumerate(retrieved_titles) if keyword.lower() in t.lower()]
                if found_ranks:
                    hits += 1
                    rr += 1.0 / found_ranks[0]
                total += 1
            hit_rate = hits / total
            mrr = rr / total
            results.append({"Method": method, "HIT@K": round(hit_rate,3), "MRR": round(mrr,3)})

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Method"))

