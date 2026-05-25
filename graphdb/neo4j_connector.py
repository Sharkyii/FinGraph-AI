import os
from typing import Optional

from neo4j import GraphDatabase, Driver

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_query(cypher: str, params: Optional[dict] = None) -> list[dict]:
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, **(params or {}))
        return [dict(record) for record in result]


def is_connected() -> bool:
    try:
        get_driver().verify_connectivity()
        return True
    except Exception as e:
        print(f"[Neo4j] Connection failed: {e}")
        return False


def create_constraints():
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Product) REQUIRE pr.name IS UNIQUE",
    ]
    for cypher in constraints:
        try:
            run_query(cypher)
        except Exception as e:
            print(f"[Neo4j] Constraint warning: {e}")
