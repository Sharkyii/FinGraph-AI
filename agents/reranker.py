from models.state import AgentState
from utils.embedding import rerank

TOP_K_RERANKED = 6


def run_reranker(state: AgentState) -> AgentState:
    if not state.docs:
        print("[Reranker] No docs to rerank.")
        state.reranked = []
        return state

    reranked = rerank(f"{state.company} {state.query}", state.docs, top_k=TOP_K_RERANKED)

    print(f"[Reranker] Top {len(reranked)} docs selected.")
    for i, doc in enumerate(reranked):
        print(f"  [{i+1}] score={doc.get('rerank_score', 0):.3f} source={doc.get('metadata', {}).get('source', 'unknown')}")

    state.reranked = reranked
    return state
