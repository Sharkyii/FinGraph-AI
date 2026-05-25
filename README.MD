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

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Tavily free API key (https://tavily.com)
- Neo4j Community Edition (optional, for graph features)

---

## Installation

### 1. Clone and set up Python environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and start Ollama, pull the model

```bash
# Install Ollama from https://ollama.com
ollama pull qwen3:8b
ollama serve   # starts on http://localhost:11434
```

### 3. Configure environment variables

Edit `.env`:

```env
TAVILY_API_KEY=your_tavily_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
QDRANT_PATH=./qdrant.db
QDRANT_COLLECTION=financial_docs
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

### 4. (Optional) Start Neo4j

```bash
# Docker:
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:community
```

### 5. Ingest financial data for a company

```bash
# Basic ingest (Qdrant only)
python ingest.py --company nvidia --quarters 2

# With Neo4j graph building
python ingest.py --company nvidia --quarters 2 --graph

# Other examples
python ingest.py --company apple --quarters 4
python ingest.py --company tesla --quarters 2
```

### 6. Start the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"company": "Nvidia", "query": "Why did guidance change?"}'
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"company": "Tesla", "query": "What risks increased in earnings?"}'
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"company": "Apple", "query": "Compare Q1 vs Q2 performance"}'
```

### Response structure

```json
{
  "company": "Nvidia",
  "query": "Why did guidance change?",
  "answer": "## Highlights\n...",
  "sentiment": {
    "overall_sentiment": "bullish",
    "confidence": 0.87,
    ...
  },
  "comparison": {
    "guidance_changes": "...",
    "risk_changes": "...",
    ...
  },
  "metrics": {
    "revenue": 26000000000,
    "eps": 5.16,
    ...
  },
  "plan": {...},
  "sources": [...],
  "errors": [],
  "elapsed_seconds": 18.4
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

### POST /ingest (API trigger)

```bash
curl -X POST "http://localhost:8000/ingest?company=meta&graph=false"
```

---

## Example Queries

- "What changed in Nvidia this quarter?"
- "Why did Tesla guidance change?"
- "Compare Apple Q1 vs Q2"
- "What risks increased in Amazon earnings?"
- "Summarize Meta earnings and explain sentiment changes"

---

## Project Structure

```
financial-agent/
├── app.py                  # FastAPI application
├── ingest.py               # Data ingestion pipeline
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
│
├── agents/
│   ├── planner.py          # Query analysis and plan generation
│   ├── retriever.py        # Tavily + Qdrant document retrieval
│   ├── reranker.py         # Cross-encoder reranking
│   ├── temporal.py         # Quarter-over-quarter comparison
│   ├── sentiment.py        # Financial sentiment analysis
│   └── synthesizer.py      # Final answer synthesis
│
├── graph/
│   └── flow.py             # LangGraph workflow definition
│
├── models/
│   └── state.py            # AgentState Pydantic model
│
├── utils/
│   ├── embedding.py        # Embedding + reranking utilities
│   ├── loaders.py          # Document chunking utilities
│   └── metrics.py          # Financial metrics extraction
│
├── graphdb/
│   ├── builder.py          # LLM-based graph extraction + insertion
│   └── neo4j_connector.py  # Neo4j connection management
│
├── vector_db/              # Qdrant storage (auto-created)
├── data/                   # Raw document storage
└── notebooks/              # Jupyter exploration notebooks
```
