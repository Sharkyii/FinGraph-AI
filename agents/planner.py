import json
import os
import re

import ollama

from models.state import AgentState

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

PLANNER_SYSTEM = """You are a financial research planning agent.
Given a user query about a company, produce a JSON execution plan.

Return ONLY valid JSON with these exact keys:
{
  "task": "<one of: earnings_summary, guidance_change, risk_analysis, comparison, sentiment, metrics, general>",
  "retrieve": ["<list of: earnings, news, guidance, annual_report, quarterly_report>"],
  "history_quarters": <integer 1-4>,
  "needs_comparison": <true or false>,
  "needs_sentiment": <true or false>,
  "needs_metrics": <true or false>,
  "focus_areas": ["<list of specific topics to focus on>"]
}

Do not include any explanation or markdown. Return raw JSON only."""


def run_planner(state: AgentState) -> AgentState:
    prompt = f"Company: {state.company}\nQuery: {state.query}"

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    state.plan = json.loads(raw)
    print(f"[Planner] Plan: {state.plan}")

    return state
