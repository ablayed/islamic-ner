"""Knowledge-graph build and query API routes."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from api.routes.ner import run_ner_pipeline
from api.schemas import Entity, GraphBuildResponse, NERRequest, Relation, ScholarQueryResponse

router = APIRouter(tags=["Graph"])


def _safe_confidence(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def _entity_from_dict(payload: Dict[str, Any]) -> Entity:
    return Entity(
        text=str(payload.get("text", "")),
        type=str(payload.get("type", "")),
        start=int(payload.get("start", -1)),
        end=int(payload.get("end", -1)),
        confidence=_safe_confidence(payload.get("confidence", 1.0), default=1.0),
    )


def _relation_from_dict(payload: Dict[str, Any]) -> Relation:
    return Relation(
        type=str(payload.get("type", "")),
        source=_entity_from_dict(payload.get("source", {})),
        target=_entity_from_dict(payload.get("target", {})),
        confidence=_safe_confidence(payload.get("confidence", 0.0)),
        evidence=str(payload.get("evidence", "")),
    )


def _count_narrated_hadiths(graph_builder: Any, scholar_canonical_name: str) -> int:
    if graph_builder is None:
        return 0

    with graph_builder.driver.session() as session:
        if hasattr(session, "count_scholar_hadiths"):
            return int(session.count_scholar_hadiths(scholar_canonical_name))

        try:
            result = session.run(
                """
                MATCH (s:Scholar {canonical_name: $name})-[r:NARRATED_FROM]-(:Scholar)
                WHERE coalesce(r.source_hadith, '') <> ''
                RETURN count(DISTINCT r.source_hadith) AS narrated_hadiths
                """,
                name=scholar_canonical_name,
            )
            record = result.single()
            if record is None:
                return 0
            return int(record.get("narrated_hadiths", 0))
        except Exception:
            return 0


@router.post("/graph/build", response_model=GraphBuildResponse)
async def build_graph(request: NERRequest, fastapi_request: Request):
    """
    Extract entities and relations, insert into knowledge graph.

    Pipeline: NER -> Relation Extraction -> Neo4j insertion
    Returns entities, relations, and insertion counts.
    """

    state = fastapi_request.app.state
    pipeline = run_ner_pipeline(
        text=request.text,
        model=getattr(state, "model", None),
        tokenizer=getattr(state, "tokenizer", None),
        normalizer=state.normalizer,
        gazetteer_matcher=getattr(state, "gazetteer_matcher", None),
    )

    metadata = {"hadith_id": "CURRENT_HADITH"}
    relation_extractor = state.relation_extractor
    raw_relations: List[Dict[str, Any]] = relation_extractor.extract(
        pipeline.words,
        pipeline.labels,
        metadata=metadata,
    )
    relation_models = [_relation_from_dict(relation) for relation in raw_relations]

    nodes_inserted = 0
    relations_inserted = 0
    graph_builder = getattr(state, "kg_builder", None)
    if graph_builder is not None:
        try:
            result = graph_builder.process_hadith(
                tokens=pipeline.words,
                labels=pipeline.labels,
                hadith_id="CURRENT_HADITH",
                metadata=metadata,
            )
            nodes_inserted = int(result.get("entities_inserted", 0))
            relations_inserted = int(result.get("relations_inserted", 0))
            state.neo4j_connected = True
        except Exception:
            # Keep API usable even if Neo4j is down; caller still gets NER + RE output.
            state.neo4j_connected = False

    return GraphBuildResponse(
        text=request.text,
        entities=pipeline.entities,
        relations=relation_models,
        nodes_inserted=nodes_inserted,
        relations_inserted=relations_inserted,
    )


@router.get("/graph/query/{scholar_name}", response_model=ScholarQueryResponse)
async def query_scholar(scholar_name: str, fastapi_request: Request):
    """
    Query the knowledge graph for a scholar's connections.
    Returns teachers, students, and hadith count.
    """

    state = fastapi_request.app.state
    querier = getattr(state, "graph_querier", None)
    graph_builder = getattr(state, "kg_builder", None)
    if querier is None or graph_builder is None:
        raise HTTPException(status_code=503, detail="Knowledge graph is not available.")

    scholar = querier.find_scholar(scholar_name)
    if not scholar:
        return ScholarQueryResponse(
            scholar={"canonical_name": scholar_name, "name_ar": scholar_name},
            teachers=[],
            students=[],
            narrated_hadiths=0,
        )

    canonical_name = str(scholar.get("canonical_name", scholar_name))
    connections = querier.get_scholar_connections(canonical_name)

    teachers = [{"name": teacher_name} for teacher_name in connections.get("teachers", [])]
    students = [{"name": student_name} for student_name in connections.get("students", [])]
    narrated_hadiths = _count_narrated_hadiths(graph_builder, canonical_name)

    return ScholarQueryResponse(
        scholar=scholar,
        teachers=teachers,
        students=students,
        narrated_hadiths=narrated_hadiths,
    )
