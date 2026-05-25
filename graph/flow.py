from langgraph.graph import StateGraph, END

from agents.planner import run_planner
from agents.retriever import run_retriever
from agents.reranker import run_reranker
from agents.temporal import run_temporal
from agents.sentiment import run_sentiment
from agents.synthesizer import run_synthesizer
from models.state import AgentState


def _to_dict(state: AgentState) -> dict:
    return state.model_dump()


def _to_state(d: dict) -> AgentState:
    return AgentState(**d)


def planner_node(state: dict) -> dict:
    return _to_dict(run_planner(_to_state(state)))


def retriever_node(state: dict) -> dict:
    return _to_dict(run_retriever(_to_state(state)))


def reranker_node(state: dict) -> dict:
    return _to_dict(run_reranker(_to_state(state)))


def temporal_node(state: dict) -> dict:
    return _to_dict(run_temporal(_to_state(state)))


def sentiment_node(state: dict) -> dict:
    return _to_dict(run_sentiment(_to_state(state)))


def synthesizer_node(state: dict) -> dict:
    return _to_dict(run_synthesizer(_to_state(state)))


def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("temporal", temporal_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "temporal")
    graph.add_edge("temporal", "sentiment")
    graph.add_edge("sentiment", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(company: str, query: str) -> AgentState:
    graph = get_graph()
    initial = AgentState(company=company.lower(), query=query)
    result = graph.invoke(_to_dict(initial))
    return AgentState(**result)
