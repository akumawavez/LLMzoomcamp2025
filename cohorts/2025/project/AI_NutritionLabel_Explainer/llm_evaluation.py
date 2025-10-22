"""
LLM Evaluation Module using RAGAS
=================================

This module implements comprehensive LLM evaluation using RAGAS metrics
to compare multiple prompt templates for the AI Nutrition Label Explainer.

Metrics evaluated:
- Faithfulness: Answer consistency with retrieved context
- Answer Relevancy: How relevant the answer is to the question  
- Context Precision: Precision of retrieved contexts
- Context Recall: Coverage of relevant information
"""

import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# RAGAS imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# LangChain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "openfoodfacts_filtered"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    prompt_name: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    avg_score: float
    sample_answers: List[str]

class LLMEvaluator:
    """Main class for LLM evaluation using RAGAS"""
    
    def __init__(self):
        """Initialize the evaluator with models and clients"""
        logger.info("="*60)
        logger.info("Initializing LLM Evaluator")
        logger.info("="*60)
        
        logger.info(f"Connecting to Ollama at {OLLAMA_URL}")
        logger.info(f"Chat Model: {CHAT_MODEL}")
        logger.info(f"Embedding Model: {EMBED_MODEL}")
        
        self.llm = ChatOllama(model=CHAT_MODEL, temperature=0)
        logger.info("✓ LLM initialized successfully")
        
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        logger.info("✓ Embeddings initialized successfully")
        
        logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info(f"✓ Qdrant client connected (collection: {COLLECTION})")
        
        # Define prompt templates
        self.prompt_templates = {
            "Current Prompt": self._get_current_prompt(),
            "Structured Expert": self._get_structured_expert_prompt(),
            "Conversational Assistant": self._get_conversational_prompt(),
            "Fact-Based Analytical": self._get_fact_based_prompt()
        }
        logger.info(f"✓ Loaded {len(self.prompt_templates)} prompt templates")
        logger.info("Evaluator initialization complete\n")
    
    def _get_current_prompt(self) -> str:
        """Current baseline prompt"""
        return """You are a nutrition expert. 
Answer the user's question using ONLY the following context (Open Food Facts product data). 
Cite product names if relevant.

Context:
{context}

Question: {question}"""
    
    def _get_structured_expert_prompt(self) -> str:
        """Structured expert prompt with clear sections"""
        return """You are a certified nutritionist analyzing food products.
Provide a structured answer with:
1. Direct answer to the question
2. Supporting evidence from the data
3. Relevant product recommendations

Use ONLY the provided context (Open Food Facts product data). Cite specific products when relevant.

Context:
{context}

Question: {question}"""
    
    def _get_conversational_prompt(self) -> str:
        """Conversational and friendly prompt"""
        return """You're a friendly nutrition assistant helping someone understand food labels.
Answer in simple, conversational language. Break down complex nutrition facts.
Use ONLY the context provided. Cite specific products when relevant.

Context:
{context}

Question: {question}"""
    
    def _get_fact_based_prompt(self) -> str:
        """Fact-based analytical prompt"""
        return """Provide a fact-based nutritional analysis.
State numerical values where available.
Compare products quantitatively when relevant.
Answer based ONLY on the context provided.

Context:
{context}

Question: {question}"""
    
    def load_test_dataset(self, file_path: str) -> List[Dict]:
        """Load test dataset from JSON file"""
        logger.info(f"Loading test dataset from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded {len(data)} test questions")
        return data
    
    def embed_text(self, text: str) -> List[float]:
        """Helper to embed a single text query"""
        return self.embeddings.embed_query(text)
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve documents using hybrid search + reranking (best method)"""
        # Embedding
        start_time = time.time()
        query_vector = self.embed_text(query)
        embed_time = time.time() - start_time
        
        # Hybrid retrieval
        start_time = time.time()
        hits = self.qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=k,
        )
        search_time = time.time() - start_time
        
        results = [
            {
                "id": h.id,
                "score": h.score,
                "payload": h.payload,
            }
            for h in hits
        ]
        
        logger.debug(f"  Retrieved {len(results)} docs (embed: {embed_time:.2f}s, search: {search_time:.2f}s)")
        
        # LLM-based reranking
        start_time = time.time()
        reranked = self._rerank_with_llm(query, results)
        rerank_time = time.time() - start_time
        logger.debug(f"  Reranked documents ({rerank_time:.2f}s)")
        
        return reranked
    
    def _rerank_with_llm(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Lightweight reranker using LLM"""
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
        
        resp = self.llm.invoke(prompt).content
        order = [int(x.strip()) for x in resp.split(",") if x.strip().isdigit()]
        
        ranked = [docs[i - 1] for i in order if 0 < i <= len(docs)]
        return ranked + [d for d in docs if d not in ranked]
    
    def generate_answer(self, query: str, context: str, prompt_template: str) -> str:
        """Generate answer using specified prompt template"""
        prompt = prompt_template.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        return response.content
    
    def prepare_evaluation_data(self, test_data: List[Dict]) -> Tuple[List[str], List[List[str]], List[str]]:
        """Prepare data for RAGAS evaluation"""
        logger.info("\nPreparing evaluation data (retrieving documents for all questions)...")
        
        questions = []
        contexts = []
        answers = []
        
        total = len(test_data)
        for idx, item in enumerate(test_data, 1):
            # Retrieve documents for each question
            logger.info(f"[{idx}/{total}] Retrieving docs for: '{item['question'][:60]}...'")
            docs = self.retrieve_documents(item["question"], k=5)
            context_texts = [d["payload"].get("text", "") for d in docs]
            
            questions.append(item["question"])
            contexts.append(context_texts)
            answers.append("")  # Will be filled by each prompt variant
        
        logger.info(f"✓ Retrieved documents for all {total} questions\n")
        return questions, contexts, answers
    
    def evaluate_prompt_template(self, prompt_name: str, prompt_template: str, 
                               test_data: List[Dict]) -> EvaluationResult:
        """Evaluate a single prompt template using RAGAS"""
        logger.info("="*60)
        logger.info(f"Evaluating Prompt Template: {prompt_name}")
        logger.info("="*60)
        
        eval_start = time.time()
        
        questions, contexts, _ = self.prepare_evaluation_data(test_data)
        
        # Generate answers for this prompt template
        logger.info(f"\nGenerating answers using '{prompt_name}' prompt...")
        answers = []
        sample_answers = []
        
        total = len(questions)
        for i, (question, context_list) in enumerate(zip(questions, contexts), 1):
            logger.info(f"[{i}/{total}] Generating answer for: '{question[:60]}...'")
            start_time = time.time()
            
            context_text = "\n\n".join(context_list)
            answer = self.generate_answer(question, context_text, prompt_template)
            answers.append(answer)
            
            gen_time = time.time() - start_time
            logger.debug(f"  Answer generated ({gen_time:.2f}s, {len(answer)} chars)")
            
            # Store first 3 answers as samples
            if i < 3:
                sample_answers.append(answer)
        
        logger.info(f"✓ Generated all {total} answers")
        
        # Prepare dataset for RAGAS
        logger.info("\nPreparing dataset for RAGAS evaluation...")
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "contexts": contexts,
            "answer": answers
        })
        logger.info(f"✓ Dataset prepared ({len(questions)} samples)")
        
        # Run RAGAS evaluation
        logger.info("\nRunning RAGAS metrics calculation...")
        logger.info("This may take several minutes depending on the LLM speed...")
        
        try:
            ragas_start = time.time()
            
            result = evaluate(
                eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            ragas_time = time.time() - ragas_start
            logger.info(f"✓ RAGAS evaluation completed ({ragas_time:.2f}s)")
            
            # Extract metrics
            faithfulness_score = result['faithfulness']
            answer_relevancy_score = result['answer_relevancy']
            context_precision_score = result['context_precision']
            context_recall_score = result['context_recall']
            
            avg_score = (faithfulness_score + answer_relevancy_score + 
                        context_precision_score + context_recall_score) / 4
            
            # Log results
            logger.info("\n" + "-"*60)
            logger.info(f"Results for '{prompt_name}':")
            logger.info(f"  Faithfulness:       {faithfulness_score:.3f}")
            logger.info(f"  Answer Relevancy:   {answer_relevancy_score:.3f}")
            logger.info(f"  Context Precision:  {context_precision_score:.3f}")
            logger.info(f"  Context Recall:     {context_recall_score:.3f}")
            logger.info(f"  Average Score:      {avg_score:.3f}")
            logger.info("-"*60)
            
            eval_total_time = time.time() - eval_start
            logger.info(f"Total evaluation time for '{prompt_name}': {eval_total_time:.2f}s\n")
            
            return EvaluationResult(
                prompt_name=prompt_name,
                faithfulness=faithfulness_score,
                answer_relevancy=answer_relevancy_score,
                context_precision=context_precision_score,
                context_recall=context_recall_score,
                avg_score=avg_score,
                sample_answers=sample_answers
            )
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {prompt_name}: {e}", exc_info=True)
            logger.warning(f"Returning default scores for {prompt_name}")
            
            # Return default scores if evaluation fails
            return EvaluationResult(
                prompt_name=prompt_name,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
                avg_score=0.0,
                sample_answers=sample_answers
            )
    
    def run_full_evaluation(self, test_data_path: str) -> List[EvaluationResult]:
        """Run evaluation on all prompt templates"""
        overall_start = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("STARTING FULL LLM EVALUATION")
        logger.info("="*60 + "\n")
        
        test_data = self.load_test_dataset(test_data_path)
        
        logger.info(f"\nEvaluating {len(self.prompt_templates)} prompt templates on {len(test_data)} questions...")
        logger.info("Prompt templates to evaluate:")
        for i, name in enumerate(self.prompt_templates.keys(), 1):
            logger.info(f"  {i}. {name}")
        logger.info("")
        
        results = []
        for idx, (prompt_name, prompt_template) in enumerate(self.prompt_templates.items(), 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Prompt {idx}/{len(self.prompt_templates)}: {prompt_name}")
            logger.info(f"{'#'*60}\n")
            
            result = self.evaluate_prompt_template(prompt_name, prompt_template, test_data)
            results.append(result)
            
            logger.info(f"✓ Completed {prompt_name}: Avg Score = {result.avg_score:.3f}\n")
        
        # Final summary
        overall_time = time.time() - overall_start
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"\nTotal evaluation time: {overall_time:.2f}s ({overall_time/60:.2f} minutes)")
        logger.info("\nFinal Rankings:")
        
        sorted_results = sorted(results, key=lambda x: x.avg_score, reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            logger.info(f"  {rank}. {result.prompt_name}: {result.avg_score:.3f}")
        
        best_result = sorted_results[0]
        logger.info(f"\n🏆 Best performing prompt: {best_result.prompt_name}")
        logger.info("="*60 + "\n")
        
        return results
    
    def get_best_prompt(self, results: List[EvaluationResult]) -> Tuple[str, str]:
        """Get the best performing prompt template"""
        best_result = max(results, key=lambda x: x.avg_score)
        return best_result.prompt_name, self.prompt_templates[best_result.prompt_name]
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to CSV"""
        logger.info(f"Saving results to {output_path}...")
        
        data = []
        for result in results:
            data.append({
                'Prompt Template': result.prompt_name,
                'Faithfulness': result.faithfulness,
                'Answer Relevancy': result.answer_relevancy,
                'Context Precision': result.context_precision,
                'Context Recall': result.context_recall,
                'Average Score': result.avg_score
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")

def main():
    """Main function for standalone evaluation"""
    logger.info("Starting standalone LLM evaluation script...")
    
    try:
        evaluator = LLMEvaluator()
        
        # Run evaluation
        test_data_path = "data_ingestion/llm_eval_test_set.json"
        results = evaluator.run_full_evaluation(test_data_path)
        
        # Get best prompt
        best_name, best_template = evaluator.get_best_prompt(results)
        logger.info(f"\n🏆 Best performing prompt: {best_name}")
        logger.info(f"\nBest prompt template:\n{best_template}")
        
        # Save results
        evaluator.save_results(results, "llm_evaluation_results.csv")
        
        logger.info("\n✓ Evaluation script completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation script failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
