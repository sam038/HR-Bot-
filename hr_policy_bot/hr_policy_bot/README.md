# HR Policy Bot — Agentic AI Capstone 2026

**Student:** [Your Name] | **Roll No:** [Your Roll No] | **Batch:** KIIT IT B.Tech  
**Course:** Agentic AI | **Instructor:** Dr. Kanthi Kiran Sirra

---

## Problem Statement

**Domain:** HR Policy Bot  
**User:** Company employees (TechCorp)  
**Problem:** Employees repeatedly ask HR the same questions about leave, payroll, benefits, and conduct — consuming HR staff time. A 24/7 intelligent assistant is needed that answers from the company handbook accurately, never hallucinating.  
**Success:** Agent answers 10 domain questions with faithfulness > 0.7 and correctly declines out-of-scope queries.  
**Tool:** `datetime` — used to provide current date/time for leave calculation context.

---

## Architecture

```
User Question
     ↓
[memory_node]    → sliding window (last 6 msgs), extract employee name
     ↓
[router_node]    → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node / tool_node / skip_node]
     ↓
[answer_node]    → grounded answer from context only
     ↓
[eval_node]      → faithfulness 0.0–1.0 → retry if < 0.7 (max 2 retries)
     ↓
[save_node]      → append to message history → END
```

## Tech Stack

- **LangGraph** — StateGraph with 8 nodes
- **ChromaDB** — Vector database (10 HR policy documents)
- **SentenceTransformer** — `all-MiniLM-L6-v2` for embeddings
- **Groq / LLaMA-3.3-70b** — LLM for routing, answering, evaluation
- **MemorySaver** — Persistent multi-turn memory via thread_id
- **Streamlit** — Web UI deployment

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
export GROQ_API_KEY="your-key-here"

# 3. Run tests
python agent.py

# 4. Launch Streamlit UI
streamlit run capstone_streamlit.py

# 5. Run RAGAS evaluation
python ragas_eval.py
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Complete agent: KB, State, 8 nodes, graph, test runner |
| `capstone_streamlit.py` | Streamlit web UI |
| `ragas_eval.py` | RAGAS baseline evaluation (5 QA pairs) |
| `requirements.txt` | Python dependencies |

## 6 Mandatory Capabilities

| # | Capability | Implementation |
|---|-----------|----------------|
| 1 | LangGraph StateGraph (3+ nodes) | 8 nodes: memory, router, retrieve, skip, tool, answer, eval, save |
| 2 | ChromaDB RAG (10+ docs) | 10 HR policy documents, each 100–300 words on one topic |
| 3 | MemorySaver + thread_id | Multi-turn memory across invoke() calls |
| 4 | Self-reflection eval node | Faithfulness scoring, retry if < 0.7, MAX_EVAL_RETRIES=2 |
| 5 | Tool use beyond retrieval | `datetime` tool for current date/time |
| 6 | Streamlit deployment | Full UI with @st.cache_resource, st.session_state |
