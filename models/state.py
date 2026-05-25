from typing import Any
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    query: str = ""
    company: str = ""
    plan: dict[str, Any] = Field(default_factory=dict)
    docs: list[dict[str, Any]] = Field(default_factory=list)
    reranked: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    sentiment: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
