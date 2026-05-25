import os
from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    print(f"[Embedding] Loading {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    print(f"[Reranker] Loading {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()


def rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    if not docs:
        return []

    reranker = get_reranker_model()
    pairs = [[query, doc.get("content", "")] for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]

    result = []
    for score, doc in scored:
        enriched = dict(doc)
        enriched["rerank_score"] = float(score)
        result.append(enriched)

    return result
