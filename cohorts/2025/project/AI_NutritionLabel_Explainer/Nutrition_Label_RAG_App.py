"""
AI Nutrition Label Explainer – Streamlit RAG App
------------------------------------------------
Tabs:
- 💬 Ask Nutrition Assistant: Chat interface powered by Ollama + Qdrant.
- 🍽 Data Ingestion: Runs ingestion of Open Food Facts data (.csv.gz).

AI Nutrition Label Explainer – LangChain + Streamlit version
Local, no FastAPI. Includes evaluation of retrieval methods.
"""

import os
import time
import subprocess
import pandas as pd
import streamlit as st
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText


## For Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# from langchain_community.memory import ConversationBufferMemory

st.set_page_config(page_title="AI Nutrition Label Explainer", page_icon="🥗", layout="wide")

import sys
# st.write("🐍 Using Python executable:", sys.executable)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
# from langchain.vectorstores import Qdrant
# from langchain.chains import RetrievalQA
from sklearn.metrics import label_ranking_average_precision_score

# Import LLM evaluation module
from llm_evaluation import LLMEvaluator

# Import feedback manager
from feedback_manager import FeedbackManager

# =============================================================================
# --- Configuration ---
# =============================================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "openfoodfacts_filtered"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# LangChain models
llm = ChatOllama(model=CHAT_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize feedback manager
feedback_manager = FeedbackManager()

# =============================================================================
# --- Utility: LangChain Retriever Wrappers ---
# =============================================================================


def embed_text(text: str) -> List[float]:
    """Helper to embed a single text query using the Ollama embedding model."""
    return embeddings.embed_query(text)


# def make_dense_retriever() -> BaseRetriever:
#     """Plain dense retrieval using embeddings only."""
#     vectordb = Qdrant(client=qdrant_client, collection_name=COLLECTION, embeddings=embeddings)
#     return vectordb.as_retriever(search_kwargs={"k": 5})

def make_dense_retriever(query: str, k: int = 5) -> List[Dict]:
    """Retrieve top-k documents from Qdrant using dense embeddings only."""
    query_vector = embed_text(query)

    hits = qdrant_client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=k,
    )

    results = [
        {
            "id": h.id,
            "score": h.score,
            "payload": h.payload,
        }
        for h in hits
    ]
    return results



# def make_hybrid_retriever(category: str | None = None, brand: str | None = None) -> BaseRetriever:
#     """Dense retrieval constrained by metadata (hybrid)."""
#     vectordb = Qdrant(client=qdrant_client, collection_name=COLLECTION, embeddings=embeddings)
#     filter_conds = []
#     if category:
#         filter_conds.append(FieldCondition(key="categories", match=MatchText(text=category)))
#     if brand:
#         filter_conds.append(FieldCondition(key="brands", match=MatchText(text=brand)))
#     if filter_conds:
#         flt = Filter(must=filter_conds)
#         return vectordb.as_retriever(search_kwargs={"k": 5, "filter": flt})
#     return vectordb.as_retriever(search_kwargs={"k": 5})

def make_hybrid_retriever(query: str, category: str | None = None, brand: str | None = None, k: int = 5) -> List[Dict]:
    """Retrieve top-k documents using dense vector search + optional metadata filters."""
    query_vector = embed_text(query)

    filters = []
    if category:
        filters.append(models.FieldCondition(key="categories", match=models.MatchText(text=category)))
    if brand:
        filters.append(models.FieldCondition(key="brands", match=models.MatchText(text=brand)))

    flt = models.Filter(must=filters) if filters else None

    hits = qdrant_client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=k,
        query_filter=flt,
    )

    results = [
        {
            "id": h.id,
            "score": h.score,
            "payload": h.payload,
        }
        for h in hits
    ]
    return results



# def rerank_with_llm(query: str, docs: List[Document]) -> List[Document]:
#     """Optional lightweight LLM reranker."""
#     if not docs:
#         return []
#     text_block = "\n".join([f"[{i}] {d.page_content[:200]}" for i, d in enumerate(docs, 1)])
#     prompt = f"""Re-rank the following documents for the query:
# Query: {query}
# Docs:
# {text_block}
# Output a comma-separated list of document indices (best to worst).
# """
#     resp = llm.invoke(prompt).content
#     order = [int(x.strip()) for x in resp.split(",") if x.strip().isdigit()]
#     ranked = [docs[i-1] for i in order if 0 < i <= len(docs)]
#     return ranked + [d for d in docs if d not in ranked]

def rerank_with_llm(query: str, docs: List[Dict]) -> List[Dict]:
    """Lightweight reranker using LLM (no LangChain retriever objects)."""
    if not docs:
        return []

    text_block = "\n".join(
        [f"[{i}] {d['payload'].get('text', '')[:200]}" for i, d in enumerate(docs, 1)]
    )

    prompt = f"""Re-rank the following documents for the query:
Query: {query}
Docs:
{text_block}
Output a comma-separated list of document indices (best to worst)."""

    resp = llm.invoke(prompt).content
    order = [int(x.strip()) for x in resp.split(",") if x.strip().isdigit()]

    ranked = [docs[i - 1] for i in order if 0 < i <= len(docs)]
    return ranked + [d for d in docs if d not in ranked]


# =============================================================================
# --- Tabs ---
# =============================================================================
tab_chat, tab_ingest, tab_eval, tab_llm_eval, tab_monitoring = st.tabs(["💬 Chat Assistant", "🍽 Data Ingestion", "📊 Retrieval Evaluation", "🎯 LLM Evaluation", "📊 Monitoring Dashboard"])

# -------------------------------------------------------------------------
# 💬 CHAT TAB
# -------------------------------------------------------------------------
with tab_chat:
    st.title("🥗 AI Nutrition Label Explainer (LangChain + Qdrant)")
    
    # Show current prompt template status
    if "best_prompt_template" in st.session_state:
        st.success("✨ Using optimized prompt template from LLM evaluation")
    else:
        st.info("💡 Run LLM evaluation to optimize prompt template")

    category = st.text_input("Category filter (optional)")
    brand = st.text_input("Brand filter (optional)")
    mode = st.selectbox("Retrieval Mode", ["Dense", "Hybrid", "Hybrid + Re-rank"])

    if "history" not in st.session_state:
        st.session_state.history = []

    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # query = st.chat_input("Ask a question about a product or nutrient...")
    
    bottom_placeholder = st._bottom.empty()

    # create the bar where we can type messages
    query = bottom_placeholder.chat_input("Ask a question about a product or nutrient...")

    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Build retriever according to mode

        # retriever = make_dense_retriever() if mode == "Dense" else make_hybrid_retriever(category, brand)
        # docs = retriever.get_relevant_documents(query)
        # if mode == "Hybrid + Re-rank":
        #     docs = rerank_with_llm(query, docs)

        # context_text = "\n\n".join([d.page_content for d in docs])

        # Retrieve documents directly from Qdrant
        if mode == "Dense":
            docs = make_dense_retriever(query, k=5)
        elif mode == "Hybrid":
            docs = make_hybrid_retriever(query, category=category, brand=brand, k=5)
        else:  # Hybrid + Re-rank
            docs = make_hybrid_retriever(query, category=category, brand=brand, k=5)
            docs = rerank_with_llm(query, docs)

        # Combine context text from payloads
        context_text = "\n\n".join([d["payload"].get("text", "") for d in docs])


        # Use best prompt template if available, otherwise use default
        if "best_prompt_template" in st.session_state:
            prompt_template = st.session_state.best_prompt_template
        else:
            prompt_template = """You are a nutrition expert. 
Answer the user's question using ONLY the following context (Open Food Facts product data). 
Cite product names if relevant.

Context:
{context}

Question: {question}"""
        
        prompt = prompt_template.format(context=context_text, question=query)
        t0 = time.time()
        answer = llm.invoke(prompt).content
        latency = round((time.time() - t0) * 1000, 1)

        # Generate unique interaction ID
        interaction_id = str(uuid.uuid4())
        
        # Save interaction data
        feedback_manager.save_interaction(
            interaction_id=interaction_id,
            query=query,
            answer=answer,
            context=context_text,
            method=mode,
            latency=latency,
            num_docs=len(docs)
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Latency: {latency} ms, Retrieved docs: {len(docs)}")
            
            # Add feedback UI
            st.markdown("---")
            st.markdown("**Was this response helpful?**")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                thumbs_up = st.button("👍", key=f"thumbs_up_{interaction_id}", help="Thumbs up")
                if thumbs_up:
                    feedback_manager.save_user_feedback(interaction_id, thumbs_up=True)
                    st.success("Thank you for your feedback!")
            
            with col2:
                thumbs_down = st.button("👎", key=f"thumbs_down_{interaction_id}", help="Thumbs down")
                if thumbs_down:
                    feedback_manager.save_user_feedback(interaction_id, thumbs_up=False)
                    st.success("Thank you for your feedback!")
            
            with col3:
                star_rating = st.selectbox(
                    "Rate this response (1-5 stars):",
                    options=["", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                    key=f"stars_{interaction_id}",
                    help="Rate the quality of this response"
                )
                if star_rating and star_rating != "":
                    rating_value = len(star_rating)
                    feedback_manager.save_user_feedback(interaction_id, star_rating=rating_value)
                    st.success(f"Thank you! Rated {rating_value} stars!")
        
        st.session_state.history.append({"role": "assistant", "content": answer})

        # with st.expander("🔍 Retrieved Context"):
        #     for d in docs:
        #         st.markdown(f"- **{d.metadata.get('product_name','Unknown')}** — {d.metadata.get('categories','')}")
        #         st.caption(d.page_content[:300] + "...")

        with st.expander("🔍 Retrieved Context"):
            for d in docs:
                payload = d["payload"]
                st.markdown(f"- **{payload.get('product_name', 'Unknown')}** — {payload.get('categories', '')}")
                st.caption(payload.get("text", "")[:300] + "...")

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
                    # retriever = make_dense_retriever()
                    docs = make_dense_retriever(query, k=k)
                elif method == "Hybrid":
                    # retriever = make_hybrid_retriever()
                    docs = make_hybrid_retriever(query, k=k)
                # docs = retriever.get_relevant_documents(query)
                # if method == "Hybrid + Re-rank":
                #     docs = rerank_with_llm(query, docs)
                else:  # Hybrid + Re-rank
                    docs = make_hybrid_retriever(query, k=k)
                    docs = rerank_with_llm(query, docs)
                # retrieved_titles = [d.metadata.get("product_name","") for d in docs[:k]]
                retrieved_titles = [d["payload"].get("product_name", "") for d in docs[:k]]
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

# -------------------------------------------------------------------------
# 🎯 LLM EVALUATION TAB
# -------------------------------------------------------------------------
with tab_llm_eval:
    st.header("🎯 LLM Evaluation with RAGAS")
    st.write("""
    Evaluate multiple prompt templates using RAGAS metrics:
    - **Faithfulness**: Answer consistency with retrieved context
    - **Answer Relevancy**: How relevant the answer is to the question
    - **Context Precision**: Precision of retrieved contexts  
    - **Context Recall**: Coverage of relevant information
    """)
    
    # Initialize evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = None
    
    if st.button("🚀 Run LLM Evaluation"):
        with st.spinner("Initializing evaluator..."):
            try:
                st.session_state.evaluator = LLMEvaluator()
                st.success("Evaluator initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize evaluator: {e}")
                st.stop()
        
        # Run evaluation
        test_data_path = os.path.join(os.path.dirname(__file__), "data_ingestion", "llm_eval_test_set.json")
        
        if not os.path.exists(test_data_path):
            st.error(f"Test dataset not found at {test_data_path}")
            st.stop()
        
        with st.spinner("Running LLM evaluation... This may take several minutes."):
            try:
                results = st.session_state.evaluator.run_full_evaluation(test_data_path)
                st.session_state.eval_results = results
                st.success("Evaluation completed successfully!")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.stop()
    
    # Display results if available
    if "eval_results" in st.session_state and st.session_state.eval_results:
        results = st.session_state.eval_results
        
        st.subheader("📊 Evaluation Results")
        
        # Create results DataFrame
        results_data = []
        for result in results:
            results_data.append({
                'Prompt Template': result.prompt_name,
                'Faithfulness': round(result.faithfulness, 3),
                'Answer Relevancy': round(result.answer_relevancy, 3),
                'Context Precision': round(result.context_precision, 3),
                'Context Recall': round(result.context_recall, 3),
                'Average Score': round(result.avg_score, 3)
            })
        
        df_results = pd.DataFrame(results_data)
        
        # Highlight best performing prompt
        best_idx = df_results['Average Score'].idxmax()
        df_display = df_results.copy()
        df_display = df_display.style.apply(
            lambda x: ['background-color: lightgreen' if x.name == best_idx else '' for _ in x], 
            axis=1
        )
        
        st.dataframe(df_display, use_container_width=True)
        
        # Display charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Metrics Comparison")
            metrics_df = df_results.set_index('Prompt Template')[['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']]
            st.bar_chart(metrics_df)
        
        with col2:
            st.subheader("🏆 Average Scores")
            avg_df = df_results.set_index('Prompt Template')[['Average Score']]
            st.bar_chart(avg_df)
        
        # Best prompt
        best_prompt = results[best_idx]
        st.success(f"🏆 **Best performing prompt**: {best_prompt.prompt_name} (Score: {best_prompt.avg_score:.3f})")
        
        # Sample answers
        st.subheader("📝 Sample Answers")
        
        for i, result in enumerate(results):
            with st.expander(f"{result.prompt_name} - Sample Answers"):
                for j, answer in enumerate(result.sample_answers):
                    st.write(f"**Sample {j+1}:**")
                    st.write(answer)
                    st.write("---")
        
        # Export results
        csv_data = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="llm_evaluation_results.csv",
            mime="text/csv"
        )
        
        # Integration option
        st.subheader("🔧 Integration")
        if st.button("✨ Use Best Prompt in Chat"):
            st.session_state.best_prompt_template = st.session_state.evaluator.prompt_templates[best_prompt.prompt_name]
            st.success(f"Best prompt ({best_prompt.prompt_name}) is now active in the Chat Assistant tab!")
    
    # Show prompt templates
    if st.expander("📋 View Prompt Templates"):
        if "evaluator" in st.session_state and st.session_state.evaluator:
            for name, template in st.session_state.evaluator.prompt_templates.items():
                st.write(f"**{name}:**")
                st.code(template)
                st.write("---")
        else:
            st.info("Run evaluation first to view prompt templates.")

# -------------------------------------------------------------------------
# 📊 MONITORING DASHBOARD TAB
# -------------------------------------------------------------------------
with tab_monitoring:
    st.header("📊 Monitoring Dashboard")
    st.write("Comprehensive analytics and user feedback insights for the AI Nutrition Label Explainer.")
    
    # Get metrics
    metrics = feedback_manager.get_metrics()
    
    # KPI Metrics Row
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Queries",
            value=metrics["total_queries"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="Avg Satisfaction",
            value=f"{metrics['avg_satisfaction']}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Avg Latency",
            value=f"{metrics['avg_latency']} ms",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Error Rate",
            value=f"{metrics['error_rate']}%",
            delta=None
        )
    
    # Additional metrics
    col5, col6 = st.columns(2)
    with col5:
        st.metric(
            label="Feedback Rate",
            value=f"{metrics['feedback_rate']}%",
            delta=None
        )
    
    with col6:
        st.metric(
            label="Avg Star Rating",
            value=f"{metrics['avg_star_rating']}/5",
            delta=None
        )
    
    # Charts Section
    st.subheader("📊 Analytics Charts")
    
    if metrics["total_queries"] == 0:
        st.info("No data available yet. Start using the chat assistant to see analytics!")
    else:
        # Create two columns for charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Chart 1: User Feedback Trends Over Time
            st.subheader("📈 Feedback Trends Over Time")
            trends_df = feedback_manager.get_feedback_trends()
            
            if not trends_df.empty:
                st.line_chart(
                    trends_df.set_index('time_group')['satisfaction_rate'],
                    use_container_width=True
                )
                st.caption("Satisfaction rate (%) over time")
            else:
                st.info("No feedback data available yet.")
            
            # Chart 3: Query Volume Metrics
            st.subheader("📊 Query Volume")
            volume_df = feedback_manager.get_query_volume()
            
            if not volume_df.empty:
                st.bar_chart(
                    volume_df.set_index('time_group')['query_count'],
                    use_container_width=True
                )
                st.caption("Number of queries over time")
            else:
                st.info("No query data available yet.")
            
            # Chart 5: Top Queried Products/Categories
            st.subheader("🏆 Top Queried Products")
            top_products_df = feedback_manager.get_top_products()
            
            if not top_products_df.empty:
                st.bar_chart(
                    top_products_df.set_index('product')['count'],
                    use_container_width=True
                )
                st.caption("Most frequently queried products")
            else:
                st.info("No product data available yet.")
        
        with col_right:
            # Chart 2: Response Latency Analysis
            st.subheader("⏱️ Response Latency Analysis")
            latency_analysis = feedback_manager.get_latency_analysis()
            
            if latency_analysis and 'stats' in latency_analysis:
                # Create a DataFrame for latency by method
                latency_df = pd.DataFrame(latency_analysis['stats']['mean']).reset_index()
                latency_df.columns = ['method', 'avg_latency']
                
                st.bar_chart(
                    latency_df.set_index('method')['avg_latency'],
                    use_container_width=True
                )
                st.caption("Average latency by retrieval method")
                
                # Show percentiles
                with st.expander("📊 Latency Percentiles"):
                    for method, percentiles in latency_analysis['percentiles'].items():
                        st.write(f"**{method}:**")
                        st.write(f"- P50: {percentiles['p50']:.1f} ms")
                        st.write(f"- P90: {percentiles['p90']:.1f} ms")
                        st.write(f"- P99: {percentiles['p99']:.1f} ms")
            else:
                st.info("No latency data available yet.")
            
            # Chart 4: Retrieval Method Usage
            st.subheader("🔍 Retrieval Method Usage")
            method_usage_df = feedback_manager.get_method_usage()
            
            if not method_usage_df.empty:
                st.bar_chart(
                    method_usage_df.set_index('method')['count'],
                    use_container_width=True
                )
                st.caption("Usage distribution by retrieval method")
                
                # Show percentages
                with st.expander("📊 Method Usage Percentages"):
                    st.dataframe(method_usage_df[['method', 'percentage']], use_container_width=True)
            else:
                st.info("No method usage data available yet.")
            
            # Chart 6: User Satisfaction Scores
            st.subheader("⭐ User Satisfaction Scores")
            satisfaction_data = feedback_manager.get_satisfaction_distribution()
            
            if satisfaction_data and 'star_distribution' in satisfaction_data:
                star_dist = satisfaction_data['star_distribution']
                if star_dist:
                    # Create DataFrame for star ratings
                    star_df = pd.DataFrame([
                        {'rating': f"{k} stars", 'count': v} 
                        for k, v in star_dist.items()
                    ])
                    
                    st.bar_chart(
                        star_df.set_index('rating')['count'],
                        use_container_width=True
                    )
                    st.caption("Distribution of star ratings")
                    
                    # Show satisfaction by method
                    if 'satisfaction_by_method' in satisfaction_data:
                        with st.expander("📊 Satisfaction by Method"):
                            method_satisfaction = satisfaction_data['satisfaction_by_method']
                            for method, rate in method_satisfaction.items():
                                st.write(f"**{method}:** {rate}% satisfaction")
                else:
                    st.info("No star rating data available yet.")
            else:
                st.info("No satisfaction data available yet.")
    
    # Raw Data Section
    with st.expander("📋 Raw Data"):
        st.subheader("Recent Interactions")
        
        df = feedback_manager.get_combined_data()
        if not df.empty:
            # Show recent interactions
            recent_df = df.tail(10)[['timestamp', 'query', 'method', 'latency', 'thumbs_up', 'star_rating']]
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            recent_df.columns = ['Timestamp', 'Query', 'Method', 'Latency (ms)', 'Thumbs Up', 'Star Rating']
            
            st.dataframe(recent_df, use_container_width=True)
            
            # Download button for full data
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Data as CSV",
                data=csv_data,
                file_name="interactions_and_feedback.csv",
                mime="text/csv"
            )
        else:
            st.info("No interaction data available yet.")
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

