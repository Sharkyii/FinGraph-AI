import argparse
import os
import re
import uuid

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tavily import TavilyClient

from utils.embedding import embed_texts
from utils.loaders import chunk_text, build_document

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant.db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_docs")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EMBEDDING_DIM = 384

SEARCH_QUERIES = [
    "{company} quarterly earnings results revenue EPS",
    "{company} annual report financial results",
    "{company} earnings call transcript guidance outlook",
    "{company} risk factors challenges headwinds",
    "{company} CEO management commentary strategy",
    "{company} latest news financial performance",
]


def _classify_source(url: str, title: str) -> str:
    combined = (url + " " + title).lower()
    if any(k in combined for k in ["transcript", "earnings call"]):
        return "earnings_transcript"
    if any(k in combined for k in ["annual report", "10-k", "10k"]):
        return "annual_report"
    if any(k in combined for k in ["10-q", "10q", "quarterly"]):
        return "quarterly_report"
    if any(k in combined for k in ["guidance", "outlook", "forecast"]):
        return "guidance"
    if any(k in combined for k in ["news", "reuters", "bloomberg", "cnbc", "wsj"]):
        return "news"
    return "financial_report"


def _extract_quarter(text: str) -> str:
    patterns = [
        r"(Q[1-4]\s*20\d{2})",
        r"(first|second|third|fourth)\s+quarter\s+20\d{2}",
        r"(FY\s*20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return "latest"


def fetch_documents(company: str, quarters: int = 2) -> list[dict]:
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set in .env")

    client = TavilyClient(api_key=TAVILY_API_KEY)
    all_docs = []

    for query_template in SEARCH_QUERIES:
        query = query_template.format(company=company)
        print(f"[Ingest] Searching: {query}")
        try:
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_raw_content=True,
            )
            for r in results.get("results", []):
                content = r.get("raw_content") or r.get("content") or ""
                if len(content) < 100:
                    continue
                all_docs.append(build_document(
                    content=content,
                    company=company.lower(),
                    source=_classify_source(r.get("url", ""), r.get("title", "")),
                    quarter=_extract_quarter(r.get("title", "") + " " + content[:200]),
                    date=r.get("published_date", "unknown"),
                    url=r.get("url", ""),
                ))
        except Exception as e:
            print(f"[Ingest] Search error for '{query}': {e}")

    print(f"[Ingest] Fetched {len(all_docs)} documents for {company}.")
    return all_docs


def setup_qdrant(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"[Ingest] Created collection: {QDRANT_COLLECTION}")
    else:
        print(f"[Ingest] Using existing collection: {QDRANT_COLLECTION}")


def ingest_documents(docs: list[dict], client: QdrantClient):
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(
            text=doc["content"],
            metadata=doc["metadata"],
            chunk_size=512,
            chunk_overlap=64,
        ))

    print(f"[Ingest] Total chunks: {len(all_chunks)}")

    batch_size = 32
    points = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        embeddings = embed_texts([c["content"] for c in batch])
        for chunk, embedding in zip(batch, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"content": chunk["content"], "metadata": chunk["metadata"]},
            ))
        print(f"[Ingest] Batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1} embedded.")

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print(f"[Ingest] Upserted {len(points)} chunks.")


def main():
    parser = argparse.ArgumentParser(description="AlphaGraph Ingest")
    parser.add_argument("--company", required=True)
    parser.add_argument("--quarters", type=int, default=2)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()

    company = args.company.lower()
    print(f"\n=== AlphaGraph Ingest: {company} ===\n")

    docs = fetch_documents(company, args.quarters)
    if not docs:
        print("[Ingest] No documents fetched. Check your Tavily API key.")
        return

    qdrant_client = QdrantClient(path=QDRANT_PATH)
    setup_qdrant(qdrant_client)
    ingest_documents(docs, qdrant_client)

    if args.graph:
        from graphdb.builder import build_graph_from_docs
        from graphdb.neo4j_connector import create_constraints
        print("\n[Ingest] Building Neo4j graph...")
        create_constraints()
        build_graph_from_docs(docs, company)

    print(f"\n[Ingest] Done. {company} is ready for querying.")


if __name__ == "__main__":
    main()
