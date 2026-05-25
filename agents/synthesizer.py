import os
import re

import ollama

from models.state import AgentState
from utils.metrics import extract_financial_metrics

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SYNTHESIS_SYSTEM = """You are a senior financial research analyst writing a structured research note.
Using the provided analysis data, write a comprehensive answer to the user's query.

Structure your response with these sections (use markdown headers):

## Highlights
Key findings and direct answer to the query.

## Financial Metrics
Revenue, EPS, margins, and growth rates found in the documents.

## Guidance Changes
Any forward guidance revisions or outlook changes.

## Risk Factors
Identified or escalated risks.

## Sentiment Analysis
Overall market and management sentiment with reasoning.

## Quarter-over-Quarter Comparison
Key differences between periods if applicable.

## Evidence & Sources
List the sources used with brief descriptions.

Be specific, cite numbers when available, and be concise. Do not hallucinate data not present in the inputs."""


def _build_prompt(state: AgentState) -> str:
    lines = [
        f"Company: {state.company}",
        f"Query: {state.query}",
        "",
        "=== PLAN ===",
        str(state.plan),
        "",
    ]

    if state.comparison:
        lines += ["=== TEMPORAL COMPARISON ===", str(state.comparison), ""]

    if state.sentiment:
        lines += ["=== SENTIMENT ANALYSIS ===", str(state.sentiment), ""]

    if state.metrics:
        lines += ["=== FINANCIAL METRICS ===", str(state.metrics), ""]

    lines.append("=== TOP EVIDENCE ===")
    for i, doc in enumerate(state.reranked[:5]):
        meta = doc.get("metadata", {})
        lines.append(f"[{i+1}] Source: {meta.get('source', 'unknown')} | Quarter: {meta.get('quarter', 'unknown')} | URL: {meta.get('url', '')}")
        lines.append(doc.get("content", "")[:600])
        lines.append("")

    return "\n".join(lines)


def run_synthesizer(state: AgentState) -> AgentState:
    if state.reranked:
        all_metrics = {}
        for doc in state.reranked[:3]:
            all_metrics.update(extract_financial_metrics(doc.get("content", "")))
        state.metrics = all_metrics

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": _build_prompt(state)},
        ],
        options={"temperature": 0.2, "num_ctx": 8192},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    state.answer = raw
    print(f"[Synthesizer] Answer generated ({len(raw)} chars).")

    return state
