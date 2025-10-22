# LLM Evaluation Logging Guide

## Overview

The `llm_evaluation.py` module now includes comprehensive logging to track evaluation progress, debug issues, and monitor performance. All logs are timestamped and use Python's standard logging module.

## Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

## What Gets Logged

### 1. Initialization Phase
- Connection to Ollama server (URL, models used)
- LLM initialization status
- Embeddings model initialization
- Qdrant client connection
- Number of prompt templates loaded

**Example Output:**
```
2025-01-22 10:30:15 - INFO - ============================================================
2025-01-22 10:30:15 - INFO - Initializing LLM Evaluator
2025-01-22 10:30:15 - INFO - ============================================================
2025-01-22 10:30:15 - INFO - Connecting to Ollama at http://localhost:11434
2025-01-22 10:30:15 - INFO - Chat Model: llama3.2:3b
2025-01-22 10:30:15 - INFO - Embedding Model: nomic-embed-text
2025-01-22 10:30:16 - INFO - ✓ LLM initialized successfully
2025-01-22 10:30:16 - INFO - ✓ Embeddings initialized successfully
2025-01-22 10:30:16 - INFO - Connecting to Qdrant at http://localhost:6333
2025-01-22 10:30:16 - INFO - ✓ Qdrant client connected (collection: openfoodfacts_filtered)
2025-01-22 10:30:16 - INFO - ✓ Loaded 4 prompt templates
2025-01-22 10:30:16 - INFO - Evaluator initialization complete
```

### 2. Test Dataset Loading
- Dataset file path
- Number of test questions loaded

**Example Output:**
```
2025-01-22 10:30:17 - INFO - Loading test dataset from: data_ingestion/llm_eval_test_set.json
2025-01-22 10:30:17 - INFO - ✓ Loaded 30 test questions
```

### 3. Document Retrieval
- Progress for each question (X/30)
- Question being processed (truncated to 60 chars)
- Timing for embedding, search, and reranking (DEBUG level)

**Example Output:**
```
2025-01-22 10:30:18 - INFO - 
Preparing evaluation data (retrieving documents for all questions)...
2025-01-22 10:30:18 - INFO - [1/30] Retrieving docs for: 'Is Corn Flakes high in sugar?...'
2025-01-22 10:30:19 - DEBUG -   Retrieved 5 docs (embed: 0.15s, search: 0.08s)
2025-01-22 10:30:20 - DEBUG -   Reranked documents (1.23s)
2025-01-22 10:30:20 - INFO - [2/30] Retrieving docs for: 'Which yogurt has less fat?...'
...
2025-01-22 10:32:45 - INFO - ✓ Retrieved documents for all 30 questions
```

### 4. Answer Generation
- Current prompt template being evaluated
- Progress for each answer (X/30)
- Question being processed
- Time taken and answer length (DEBUG level)

**Example Output:**
```
2025-01-22 10:32:46 - INFO - ============================================================
2025-01-22 10:32:46 - INFO - Evaluating Prompt Template: Current Prompt
2025-01-22 10:32:46 - INFO - ============================================================
2025-01-22 10:32:46 - INFO - 
Generating answers using 'Current Prompt' prompt...
2025-01-22 10:32:46 - INFO - [1/30] Generating answer for: 'Is Corn Flakes high in sugar?...'
2025-01-22 10:32:48 - DEBUG -   Answer generated (1.85s, 324 chars)
2025-01-22 10:32:48 - INFO - [2/30] Generating answer for: 'Which yogurt has less fat?...'
...
2025-01-22 10:35:12 - INFO - ✓ Generated all 30 answers
```

### 5. RAGAS Evaluation
- Dataset preparation status
- RAGAS metrics calculation start (with time warning)
- RAGAS completion time
- Individual metric scores
- Average score

**Example Output:**
```
2025-01-22 10:35:12 - INFO - 
Preparing dataset for RAGAS evaluation...
2025-01-22 10:35:12 - INFO - ✓ Dataset prepared (30 samples)
2025-01-22 10:35:12 - INFO - 
Running RAGAS metrics calculation...
2025-01-22 10:35:12 - INFO - This may take several minutes depending on the LLM speed...
2025-01-22 10:42:35 - INFO - ✓ RAGAS evaluation completed (443.25s)
2025-01-22 10:42:35 - INFO - 
------------------------------------------------------------
2025-01-22 10:42:35 - INFO - Results for 'Current Prompt':
2025-01-22 10:42:35 - INFO -   Faithfulness:       0.823
2025-01-22 10:42:35 - INFO -   Answer Relevancy:   0.867
2025-01-22 10:42:35 - INFO -   Context Precision:  0.745
2025-01-22 10:42:35 - INFO -   Context Recall:     0.791
2025-01-22 10:42:35 - INFO -   Average Score:      0.807
2025-01-22 10:42:35 - INFO - ------------------------------------------------------------
2025-01-22 10:42:35 - INFO - Total evaluation time for 'Current Prompt': 588.42s
```

### 6. Error Handling
- Detailed error messages with stack traces
- Fallback to default scores on failure

**Example Output:**
```
2025-01-22 10:45:12 - ERROR - ❌ Error evaluating Structured Expert: Connection timeout
Traceback (most recent call last):
  ...
2025-01-22 10:45:12 - WARNING - Returning default scores for Structured Expert
```

### 7. Final Summary
- Total evaluation time
- Rankings of all prompt templates
- Best performing prompt

**Example Output:**
```
2025-01-22 11:15:30 - INFO - 
============================================================
2025-01-22 11:15:30 - INFO - EVALUATION COMPLETE
2025-01-22 11:15:30 - INFO - ============================================================
2025-01-22 11:15:30 - INFO - 
Total evaluation time: 2715.45s (45.26 minutes)
2025-01-22 11:15:30 - INFO - 
Final Rankings:
2025-01-22 11:15:30 - INFO -   1. Structured Expert: 0.845
2025-01-22 11:15:30 - INFO -   2. Fact-Based Analytical: 0.823
2025-01-22 11:15:30 - INFO -   3. Current Prompt: 0.807
2025-01-22 11:15:30 - INFO -   4. Conversational Assistant: 0.789
2025-01-22 11:15:30 - INFO - 
🏆 Best performing prompt: Structured Expert
2025-01-22 11:15:30 - INFO - ============================================================
```

### 8. Results Saving
- File path where results are saved
- Confirmation message

**Example Output:**
```
2025-01-22 11:15:31 - INFO - Saving results to llm_evaluation_results.csv...
2025-01-22 11:15:31 - INFO - ✓ Results saved to llm_evaluation_results.csv
```

## Log Levels Used

- **INFO**: Main progress updates, milestones, results
- **DEBUG**: Detailed timing information, individual operation times
- **WARNING**: Non-critical issues, fallback behaviors
- **ERROR**: Exceptions and failures (includes stack traces)

## Viewing Logs

### In Streamlit App
Logs will appear in the terminal/console where you run the Streamlit app:
```bash
streamlit run Nutrition_Label_RAG_App.py
```

### Standalone Script
Run the evaluation script directly to see all logs:
```bash
python llm_evaluation.py
```

### Changing Log Level
To see DEBUG messages (detailed timing), modify the logging configuration:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Performance Monitoring

The logs include timing information for:
- **Embedding time**: Time to convert query to vector
- **Search time**: Time to search Qdrant
- **Rerank time**: Time for LLM-based reranking
- **Answer generation time**: Time per answer
- **RAGAS calculation time**: Time for metric computation
- **Total evaluation time**: End-to-end duration

This helps identify bottlenecks and optimize performance.

## Troubleshooting

If evaluation is slow or stuck, check the logs for:
1. **Connection issues**: Check Ollama/Qdrant connection messages
2. **Slow operations**: Look at timing logs to find bottlenecks
3. **Errors**: Check ERROR and WARNING messages
4. **Progress**: Verify questions are being processed (look for [X/30] progress)

## Benefits

1. **Transparency**: See exactly what's happening during evaluation
2. **Debugging**: Quickly identify where failures occur
3. **Performance**: Monitor operation times to optimize
4. **Progress Tracking**: Know how much is done and how much remains
5. **Reproducibility**: Logs provide audit trail of evaluation runs
