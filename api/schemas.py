"""Pydantic request/response schemas for IslamicNER API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NERRequest(BaseModel):
    """Input payload for NER and graph-build endpoints."""

    text: str = Field(..., min_length=1, description="Raw Arabic text")
    return_tokens: bool = Field(default=False, description="Optionally return token-level detail")


class Entity(BaseModel):
    text: str
    type: str = Field(..., description="SCHOLAR, BOOK, CONCEPT, PLACE, HADITH_REF")
    start: int
    end: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class Relation(BaseModel):
    type: str = Field(..., description="NARRATED_FROM, IN_BOOK, MENTIONS_CONCEPT, AUTHORED")
    source: Entity
    target: Entity
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str


class NERResponse(BaseModel):
    text: str
    normalized_text: str
    entities: List[Entity]
    tokens: Optional[List[Dict[str, Any]]] = None


class GraphBuildResponse(BaseModel):
    text: str
    entities: List[Entity]
    relations: List[Relation]
    nodes_inserted: int
    relations_inserted: int


class ScholarQueryResponse(BaseModel):
    scholar: Dict[str, Any]
    teachers: List[Dict[str, Any]]
    students: List[Dict[str, Any]]
    narrated_hadiths: int
