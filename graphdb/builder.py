import json
import os
import re
from typing import Optional

import ollama

from graphdb.neo4j_connector import run_query, is_connected

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

EXTRACTION_SYSTEM = """You are a financial knowledge graph extraction agent.
Extract entities and relationships from the provided financial text.

Return ONLY valid JSON with this structure:
{
  "entities": [
    {"type": "Company", "name": "<name>"},
    {"type": "CEO", "name": "<name>"},
    {"type": "Risk", "name": "<risk description>"},
    {"type": "Product", "name": "<product name>"},
    {"type": "RevenueSegment", "name": "<segment name>"},
    {"type": "Country", "name": "<country name>"},
    {"type": "Event", "name": "<event description>"}
  ],
  "relationships": [
    {"from_type": "CEO", "from_name": "<name>", "relation": "CEO_OF", "to_type": "Company", "to_name": "<name>"},
    {"from_type": "Company", "from_name": "<name>", "relation": "AFFECTED_BY", "to_type": "Risk", "to_name": "<risk>"},
    {"from_type": "Company", "from_name": "<name>", "relation": "HAS_PRODUCT", "to_type": "Product", "to_name": "<product>"},
    {"from_type": "Company", "from_name": "<name>", "relation": "COMPETES_WITH", "to_type": "Company", "to_name": "<competitor>"},
    {"from_type": "Company", "from_name": "<name>", "relation": "HAS_GUIDANCE", "to_type": "Event", "to_name": "<guidance event>"},
    {"from_type": "Company", "from_name": "<name>", "relation": "OPERATES_IN", "to_type": "Country", "to_name": "<country>"}
  ]
}

Only extract entities clearly mentioned in the text. Return raw JSON only."""


def extract_graph_data(text: str, company: str) -> dict:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Company context: {company}\n\nText:\n{text[:3000]}"},
        ],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    return json.loads(raw)


def _upsert_node(node_type: str, name: str, extra_props: Optional[dict] = None):
    props = {"name": name}
    if extra_props:
        props.update(extra_props)
    run_query(
        f"MERGE (n:{node_type} {{name: $name}}) SET n += $props RETURN n",
        {"name": name, "props": props},
    )


def _upsert_relationship(from_type: str, from_name: str, relation: str, to_type: str, to_name: str):
    run_query(
        f"MERGE (a:{from_type} {{name: $from_name}}) "
        f"MERGE (b:{to_type} {{name: $to_name}}) "
        f"MERGE (a)-[r:{relation}]->(b) RETURN r",
        {"from_name": from_name, "to_name": to_name},
    )


def insert_graph_data(graph_data: dict):
    for entity in graph_data.get("entities", []):
        name = entity.get("name", "").strip()
        if name:
            _upsert_node(entity.get("type", "Entity"), name)

    for rel in graph_data.get("relationships", []):
        _upsert_relationship(
            rel["from_type"], rel["from_name"],
            rel["relation"],
            rel["to_type"], rel["to_name"],
        )


def build_graph_from_docs(docs: list[dict], company: str):
    if not is_connected():
        print("[GraphBuilder] Neo4j not available, skipping.")
        return

    print(f"[GraphBuilder] Building graph for {company} from {len(docs)} docs...")
    for i, doc in enumerate(docs):
        content = doc.get("content", "")
        if len(content) < 100:
            continue
        print(f"[GraphBuilder] Doc {i+1}/{len(docs)}...")
        insert_graph_data(extract_graph_data(content, company))

    print(f"[GraphBuilder] Done for {company}.")
