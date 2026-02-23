"""FastAPI entrypoint for IslamicNER inference and graph APIs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForTokenClassification, AutoTokenizer

from api.routes.graph import router as graph_router
from api.routes.health import router as health_router
from api.routes.ner import router as ner_router
from src.graph.builder import KnowledgeGraphBuilder
from src.graph.query import GraphQuerier
from src.preprocessing.gazetteers import GazetteerMatcher
from src.preprocessing.normalize import ArabicNormalizer
from src.relations.extract import RelationExtractor

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/islamic_ner_standard/final_model")
DEFAULT_ARABERT_TOKENIZER = "aubmindlab/bert-base-arabertv02"

app = FastAPI(
    title="IslamicNER API",
    description="Arabic Islamic Text NER and Knowledge Graph API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_model_path() -> Path:
    configured = os.getenv("MODEL_PATH")
    if not configured:
        return DEFAULT_MODEL_PATH
    return Path(configured)


def _load_tokenizer(model_path: Path):
    tokenizer_source = os.getenv("ARABERT_TOKENIZER", DEFAULT_ARABERT_TOKENIZER)
    load_errors: list[str] = []

    # Prefer model-local tokenizer for reproducibility/offline usage.
    try:
        return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
    except Exception as exc:  # pragma: no cover - guarded startup path
        load_errors.append(str(exc))

    # Fallback to configured AraBERT tokenizer id/path.
    try:
        return AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    except Exception as exc:  # pragma: no cover - guarded startup path
        load_errors.append(str(exc))

    raise RuntimeError("Unable to load tokenizer. Errors: " + " | ".join(load_errors))


def _probe_neo4j(graph_builder: KnowledgeGraphBuilder) -> bool:
    try:
        with graph_builder.driver.session() as session:
            if hasattr(session, "run"):
                session.run("RETURN 1 AS ok")
        return True
    except Exception:
        return False


def _init_state_defaults() -> None:
    app.state.model = None
    app.state.tokenizer = None
    app.state.device = torch.device("cpu")
    app.state.model_path = str(_resolve_model_path())
    app.state.model_load_error = ""

    app.state.normalizer = ArabicNormalizer()
    app.state.relation_extractor = RelationExtractor()
    app.state.gazetteer_matcher = GazetteerMatcher()

    app.state.kg_builder = None
    app.state.graph_querier = None
    app.state.neo4j_connected = False
    app.state.neo4j_error = ""


@app.on_event("startup")
async def startup_event() -> None:
    """Load model + runtime services exactly once on app startup."""

    _init_state_defaults()

    skip_model_load = os.getenv("ISLAMIC_NER_SKIP_MODEL_LOAD", "0") == "1"
    if not skip_model_load:
        model_path = _resolve_model_path()
        app.state.model_path = str(model_path)
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            tokenizer = _load_tokenizer(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            app.state.model = model
            app.state.tokenizer = tokenizer
            app.state.device = device
        except Exception as exc:  # pragma: no cover - guarded startup path
            app.state.model_load_error = str(exc)
            logger.exception("Failed to load NER model/tokenizer from %s", model_path)
    else:
        app.state.model_load_error = "Model loading skipped by ISLAMIC_NER_SKIP_MODEL_LOAD=1."

    skip_neo4j = os.getenv("ISLAMIC_NER_SKIP_NEO4J", "0") == "1"
    if not skip_neo4j:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            graph_builder = KnowledgeGraphBuilder(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )
            if _probe_neo4j(graph_builder):
                app.state.kg_builder = graph_builder
                app.state.graph_querier = GraphQuerier(graph_builder.driver)
                app.state.neo4j_connected = True
            else:
                app.state.neo4j_error = "Neo4j is not reachable."
                graph_builder.close()
        except Exception as exc:  # pragma: no cover - guarded startup path
            app.state.neo4j_error = str(exc)
            logger.exception("Failed to initialize Neo4j driver")
    else:
        app.state.neo4j_error = "Neo4j initialization skipped by ISLAMIC_NER_SKIP_NEO4J=1."


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close Neo4j connection on app shutdown."""

    graph_builder: Any = getattr(app.state, "kg_builder", None)
    if graph_builder is not None:
        try:
            graph_builder.close()
        except Exception:  # pragma: no cover - guarded shutdown path
            logger.exception("Error while closing Neo4j driver")


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    return {"message": "IslamicNER API", "docs": "/docs"}


app.include_router(health_router)
app.include_router(ner_router)
app.include_router(graph_router)
