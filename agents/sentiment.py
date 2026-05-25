import json
import os
import re

import ollama

from models.state import AgentState

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

SENTIMENT_SYSTEM = """You are a financial sentiment analyst.
Analyze the provided financial documents and classify the overall sentiment.

Return ONLY valid JSON with this structure:
{
  "overall_sentiment": "<bullish | neutral | bearish>",
  "confidence": <float between 0.0 and 1.0>,
  "earnings_sentiment": "<bullish | neutral | bearish>",
  "news_sentiment": "<bullish | neutral | bearish>",
  "management_tone": "<optimistic | cautious | defensive | confident>",
  "key_positive_signals": ["<list of positive indicators>"],
  "key_negative_signals": ["<list of negative indicators or risks>"],
  "sentiment_change_vs_prior": "<improved | unchanged | deteriorated | unknown>"
}

Return raw JSON only, no markdown, no explanation."""


def _build_context(docs: list[dict], max_chars: int = 5000) -> str:
    parts = []
    total = 0
    for doc in docs:
        chunk = f"[{doc.get('metadata', {}).get('source', 'unknown')}]\n{doc.get('content', '')}\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)


def run_sentiment(state: AgentState) -> AgentState:
    if not state.reranked:
        print("[Sentiment] No docs available.")
        state.sentiment = {"overall_sentiment": "neutral", "confidence": 0.0}
        return state

    if not state.plan.get("needs_sentiment", True):
        print("[Sentiment] Not needed per plan.")
        state.sentiment = {}
        return state

    prompt = f"Company: {state.company}\nQuery: {state.query}\n\nDocuments:\n{_build_context(state.reranked)}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SENTIMENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    state.sentiment = json.loads(raw)
    print(f"[Sentiment] {state.sentiment.get('overall_sentiment')} (confidence={state.sentiment.get('confidence')})")
    return state
