import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.flow import run_pipeline, get_graph
from models.state import AgentState


class QueryRequest(BaseModel):
    company: str = Field(..., description="Company name, e.g. 'Nvidia'")
    query: str = Field(..., description="Natural language financial question")


class QueryResponse(BaseModel):
    company: str
    query: str
    answer: str
    sentiment: dict
    comparison: dict
    metrics: dict
    plan: dict
    sources: list[dict]
    errors: list[str]
    elapsed_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[AlphaGraph] Warming up...")
    try:
        get_graph()
        from utils.embedding import get_embedding_model, get_reranker_model
        get_embedding_model()
        get_reranker_model()
        print("[AlphaGraph] Ready.")
    except Exception as e:
        print(f"[AlphaGraph] Warmup warning: {e}")
    yield
    print("[AlphaGraph] Shutting down.")


app = FastAPI(
    title="AlphaGraph",
    description="Agentic Financial Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AlphaGraph"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not request.company.strip():
        raise HTTPException(status_code=400, detail="company is required")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    start = time.time()

    try:
        state: AgentState = run_pipeline(
            company=request.company.strip(),
            query=request.query.strip(),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    sources = []
    for doc in state.reranked:
        meta = doc.get("metadata", {})
        sources.append({
            "source": meta.get("source", "unknown"),
            "quarter": meta.get("quarter", "unknown"),
            "date": meta.get("date", "unknown"),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "rerank_score": doc.get("rerank_score", 0.0),
        })

    return QueryResponse(
        company=request.company,
        query=request.query,
        answer=state.answer,
        sentiment=state.sentiment,
        comparison=state.comparison,
        metrics=state.metrics,
        plan=state.plan,
        sources=sources,
        errors=state.errors,
        elapsed_seconds=round(time.time() - start, 2),
    )


@app.post("/ingest")
async def ingest_endpoint(company: str, graph: bool = False):
    from ingest import fetch_documents, ingest_documents, setup_qdrant
    from qdrant_client import QdrantClient

    try:
        docs = fetch_documents(company.lower())
        qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant.db"))
        setup_qdrant(qdrant_client)
        ingest_documents(docs, qdrant_client)

        if graph:
            from graphdb.builder import build_graph_from_docs
            from graphdb.neo4j_connector import create_constraints
            create_constraints()
            build_graph_from_docs(docs, company.lower())

        return {"status": "ok", "company": company, "docs_ingested": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
