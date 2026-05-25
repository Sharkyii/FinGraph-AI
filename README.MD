# AlphaGraph: Agentic Financial Intelligence Platform

Production-grade AI financial analyst that answers company questions by retrieving, comparing, reasoning, and synthesizing information from financial reports, earnings transcripts, news, and structured financial data.

---

## Architecture

```
User Query
    ↓
Planner Agent        (Qwen3:8b via Ollama)
    ↓
Retriever Agent      (Tavily live search + Qdrant vector DB)
    ↓
Reranker Agent       (BAAI/bge-reranker-base cross-encoder)
    ↓
Temporal Agent       (Quarter-over-quarter comparison)
    ↓
Sentiment Agent      (Bullish / Neutral / Bearish classification)
    ↓
Synthesizer Agent    (Final structured research note)
    ↓
Evidence-backed Answer
```

---

## Tech Stack

| Component       | Technology                    |
|----------------|-------------------------------|
| Backend         | FastAPI                       |
| Orchestration   | LangGraph                     |
| Local LLM       | Qwen3:8b via Ollama           |
| Embeddings      | BAAI/bge-small-en-v1.5        |
| Reranker        | BAAI/bge-reranker-base        |
| Vector DB       | Qdrant (local)                |
| Web Search      | Tavily free tier              |
| Graph DB        | Neo4j Community Edition       |
| Chunking        | LangChain RecursiveCharacterTextSplitter |

---
