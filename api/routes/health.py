"""Health and readiness routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check(request: Request):
    """Returns API status, model loaded flag, Neo4j connected flag."""

    state = request.app.state
    model_loaded = getattr(state, "model", None) is not None
    neo4j_connected = bool(getattr(state, "neo4j_connected", False))

    # Best-effort probe for current Neo4j connectivity when a builder exists.
    graph_builder = getattr(state, "kg_builder", None)
    if graph_builder is not None:
        try:
            with graph_builder.driver.session() as session:
                if hasattr(session, "run"):
                    session.run("RETURN 1 AS ok")
            neo4j_connected = True
        except Exception:
            neo4j_connected = False

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "neo4j_connected": neo4j_connected,
    }
