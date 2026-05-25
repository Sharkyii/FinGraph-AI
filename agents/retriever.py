import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from tavily import TavilyClient

from models.state import AgentState
from utils.embedding import embed_query

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant.db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_docs")
TOP_K_VECTOR = 10
TOP_K_TAVILY = 6


def _tavily_search(query: str, company: str) -> list[dict]:
    if not TAVILY_API_KEY:
        print("[Retriever] No Tavily API key, skipping web search.")
        return []
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=f"{company} {query} earnings financial results",
            search_depth="advanced",
            max_results=TOP_K_TAVILY,
            include_raw_content=True,
        )
        docs = []
        for r in results.get("results", []):
            content = r.get("raw_content") or r.get("content") or ""
            if len(content) < 50:
                continue
            docs.append({
                "content": content[:3000],
                "metadata": {
                    "company": company,
                    "source": "tavily_web",
                    "quarter": "latest",
                    "date": r.get("published_date", "unknown"),
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                },
            })
        print(f"[Retriever] Tavily returned {len(docs)} docs.")
        return docs
    except Exception as e:
        print(f"[Retriever] Tavily error: {e}")
        return []


def _qdrant_search(query: str, company: str) -> list[dict]:
    try:
        client = QdrantClient(path=QDRANT_PATH)
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            print("[Retriever] Qdrant collection not found. Run ingest.py first.")
            return []

        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embed_query(query),
            query_filter=Filter(
                must=[FieldCondition(key="metadata.company", match=MatchValue(value=company.lower()))]
            ),
            limit=TOP_K_VECTOR,
            with_payload=True,
        )

        docs = []
        for hit in response.points:
            payload = hit.payload or {}
            docs.append({
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "score": hit.score,
            })
        print(f"[Retriever] Qdrant returned {len(docs)} docs.")
        return docs
    except Exception as e:
        print(f"[Retriever] Qdrant error: {e}")
        return []


def _deduplicate(docs: list[dict], threshold: int = 200) -> list[dict]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.get("content", "")[:threshold]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def run_retriever(state: AgentState) -> AgentState:
    tavily_docs = _tavily_search(state.query, state.company)
    qdrant_docs = _qdrant_search(state.query, state.company)

    all_docs = _deduplicate(tavily_docs + qdrant_docs)
    print(f"[Retriever] Total unique docs: {len(all_docs)}")
    state.docs = all_docs
    return state
