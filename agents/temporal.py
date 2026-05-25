import json
import os
import re

import ollama

from models.state import AgentState

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

TEMPORAL_SYSTEM = """You are a financial analyst specializing in quarter-over-quarter analysis.
Given earnings documents from multiple periods, identify changes between periods.

Return ONLY valid JSON with this structure:
{
  "guidance_changes": "<description of any guidance revisions>",
  "risk_changes": "<new or escalated risks vs prior period>",
  "tone_shift": "<management tone: more bullish / neutral / bearish vs prior>",
  "strategic_changes": "<any new strategic priorities or pivots>",
  "key_differences": ["<list of bullet point differences>"],
  "quarters_compared": ["<list of quarters found in documents>"]
}

Base your analysis only on the provided documents. Return raw JSON only, no markdown."""


def _build_context(docs: list[dict], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for doc in docs:
        meta = doc.get("metadata", {})
        chunk = f"[Source: {meta.get('source', 'unknown')} | Quarter: {meta.get('quarter', 'unknown')}]\n{doc.get('content', '')}\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


def run_temporal(state: AgentState) -> AgentState:
    if not state.reranked:
        print("[Temporal] No reranked docs available.")
        state.comparison = {"error": "no documents available for comparison"}
        return state

    if not state.plan.get("needs_comparison", True):
        print("[Temporal] Comparison not needed per plan.")
        state.comparison = {}
        return state

    prompt = f"Company: {state.company}\nQuery: {state.query}\n\nDocuments:\n{_build_context(state.reranked)}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": TEMPORAL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    state.comparison = json.loads(raw)
    print("[Temporal] Comparison complete.")

    return state
