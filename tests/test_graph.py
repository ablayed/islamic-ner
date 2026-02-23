"""Tests for entity resolution and graph construction/querying."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import pytest

from src.graph.builder import KnowledgeGraphBuilder
from src.graph.entity_resolver import EntityResolver
from src.graph.query import GraphQuerier
from src.preprocessing.normalize import ArabicNormalizer


class FakeGraphBackend:
    """In-memory graph backend used as a Neo4j test double."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, dict]] = defaultdict(dict)
        self.relationships: dict[tuple, dict] = {}
        self.constraints: list[str] = []

    def clear_graph(self) -> None:
        self.nodes.clear()
        self.relationships.clear()

    def create_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)

    def merge_entity_node(
        self, label: str, key_field: str, key_value: str, properties: dict
    ) -> None:
        bucket = self.nodes[label]
        node = bucket.get(key_value, {key_field: key_value})
        if "variants" in properties:
            existing = set(node.get("variants", []))
            existing.update(properties.get("variants", []))
            node["variants"] = sorted(existing)
        for key, value in properties.items():
            if key == "variants":
                continue
            node[key] = value
        bucket[key_value] = node

    def merge_relation(
        self, relation_type: str, source_node: dict, target_node: dict, properties: dict
    ) -> None:
        self.merge_entity_node(
            source_node["label"],
            source_node["key_field"],
            source_node["key_value"],
            source_node["properties"],
        )
        self.merge_entity_node(
            target_node["label"],
            target_node["key_field"],
            target_node["key_value"],
            target_node["properties"],
        )

        key = (
            relation_type,
            source_node["label"],
            source_node["key_value"],
            target_node["label"],
            target_node["key_value"],
            str(properties.get("source_hadith", "")),
        )
        if key not in self.relationships:
            self.relationships[key] = dict(properties)
            return

        current = self.relationships[key]
        current["confidence"] = max(
            float(current.get("confidence", 0.0)),
            float(properties.get("confidence", 0.0)),
        )
        if not current.get("evidence"):
            current["evidence"] = properties.get("evidence", "")
        current["source_hadith"] = str(
            properties.get("source_hadith", current.get("source_hadith", ""))
        )
        self.relationships[key] = current

    def get_stats(self) -> dict:
        nodes_by_label = {label: len(bucket) for label, bucket in self.nodes.items()}
        relationships_by_type: dict[str, int] = {}
        for rel_key in self.relationships:
            rel_type = rel_key[0]
            relationships_by_type[rel_type] = relationships_by_type.get(rel_type, 0) + 1
        return {
            "nodes_by_label": nodes_by_label,
            "relationships_by_type": relationships_by_type,
            "total_nodes": int(sum(nodes_by_label.values())),
            "total_relationships": int(sum(relationships_by_type.values())),
        }

    def find_scholar(self, name: str) -> dict:
        candidates = []
        for canonical_name, node in self.nodes.get("Scholar", {}).items():
            name_ar = str(node.get("name_ar", canonical_name))
            if name in canonical_name or name in name_ar:
                candidates.append((canonical_name, name_ar))
        if not candidates:
            return {}
        candidates.sort(key=lambda item: len(item[0]))
        canonical_name, name_ar = candidates[0]
        return {"canonical_name": canonical_name, "name_ar": name_ar}

    def get_narration_chain(self, hadith_id: str) -> list[dict]:
        rows = []
        for key, props in self.relationships.items():
            rel_type, src_label, src_key, tgt_label, tgt_key, source_hadith = key
            if rel_type != "NARRATED_FROM":
                continue
            if source_hadith != hadith_id:
                continue
            if src_label != "Scholar" or tgt_label != "Scholar":
                continue
            rows.append(
                {
                    "source": src_key,
                    "target": tgt_key,
                    "confidence": float(props.get("confidence", 0.0)),
                    "evidence": props.get("evidence", ""),
                }
            )
        return rows

    def get_scholar_connections(self, scholar_name: str) -> dict:
        teachers = set()
        students = set()
        for key in self.relationships:
            rel_type, _, src_key, _, tgt_key, _ = key
            if rel_type != "NARRATED_FROM":
                continue
            if src_key == scholar_name:
                teachers.add(tgt_key)
            if tgt_key == scholar_name:
                students.add(src_key)
        return {
            "scholar": scholar_name,
            "teachers": sorted(teachers),
            "students": sorted(students),
        }

    def get_concepts_in_book(self, book_name: str) -> list[str]:
        hadith_ids = set()
        for key in self.relationships:
            rel_type, _, src_key, tgt_label, tgt_key, _ = key
            if rel_type != "IN_BOOK":
                continue
            if tgt_label != "Book":
                continue
            if book_name in tgt_key:
                hadith_ids.add(src_key)

        concepts = set()
        for key in self.relationships:
            rel_type, _, src_key, tgt_label, tgt_key, _ = key
            if rel_type != "MENTIONS_CONCEPT":
                continue
            if src_key in hadith_ids and tgt_label == "Concept":
                concepts.add(tgt_key)
        return sorted(concepts)

    def shortest_path(self, scholar1: str, scholar2: str) -> list[str]:
        graph: dict[str, set[str]] = defaultdict(set)
        for key in self.relationships:
            rel_type, src_label, src_key, tgt_label, tgt_key, _ = key
            if (
                rel_type != "NARRATED_FROM"
                or src_label != "Scholar"
                or tgt_label != "Scholar"
            ):
                continue
            graph[src_key].add(tgt_key)
            graph[tgt_key].add(src_key)

        if scholar1 == scholar2:
            return [scholar1]
        if scholar1 not in graph or scholar2 not in graph:
            return []

        queue = deque([(scholar1, [scholar1])])
        seen = {scholar1}
        while queue:
            node, path = queue.popleft()
            for nxt in graph[node]:
                if nxt in seen:
                    continue
                new_path = path + [nxt]
                if nxt == scholar2:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
        return []


class FakeSession:
    def __init__(self, backend: FakeGraphBackend) -> None:
        self.backend = backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def clear_graph(self) -> None:
        self.backend.clear_graph()

    def create_constraint(self, constraint: str) -> None:
        self.backend.create_constraint(constraint)

    def merge_entity_node(
        self, label: str, key_field: str, key_value: str, properties: dict
    ) -> None:
        self.backend.merge_entity_node(label, key_field, key_value, properties)

    def merge_relation(
        self, relation_type: str, source_node: dict, target_node: dict, properties: dict
    ) -> None:
        self.backend.merge_relation(relation_type, source_node, target_node, properties)

    def get_stats(self) -> dict:
        return self.backend.get_stats()

    def find_scholar(self, name: str) -> dict:
        return self.backend.find_scholar(name)

    def get_narration_chain(self, hadith_id: str) -> list[dict]:
        return self.backend.get_narration_chain(hadith_id)

    def get_scholar_connections(self, scholar_name: str) -> dict:
        return self.backend.get_scholar_connections(scholar_name)

    def get_concepts_in_book(self, book_name: str) -> list[str]:
        return self.backend.get_concepts_in_book(book_name)

    def shortest_path(self, scholar1: str, scholar2: str) -> list[str]:
        return self.backend.shortest_path(scholar1, scholar2)

    def run(self, *_args, **_kwargs):
        raise AssertionError("Cypher fallback should not run in fake-session tests")


class FakeDriver:
    def __init__(self, backend: FakeGraphBackend) -> None:
        self.backend = backend

    def session(self):
        return FakeSession(self.backend)

    def close(self) -> None:
        return None


@pytest.fixture()
def resolver() -> EntityResolver:
    fixture_dir = Path("tests/fixtures/resolver_gazetteers")
    return EntityResolver(gazetteer_dir=str(fixture_dir))


@pytest.fixture()
def graph_builder(resolver: EntityResolver):
    backend = FakeGraphBackend()
    driver = FakeDriver(backend)
    builder = KnowledgeGraphBuilder(driver=driver, resolver=resolver)
    builder.clear_graph()
    builder.create_constraints()
    yield builder, backend, resolver
    builder.close()


def test_entity_resolver_exact_match(resolver: EntityResolver) -> None:
    normalizer = ArabicNormalizer()
    result = resolver.resolve("\u0627\u0644\u0628\u062e\u0627\u0631\u064a", "SCHOLAR")
    assert result["match_type"] == "exact"
    assert result["confidence"] == 1.0
    assert result["canonical_name"] == normalizer.normalize(
        "\u0645\u062d\u0645\u062f \u0628\u0646 \u0625\u0633\u0645\u0627\u0639\u064a\u0644 \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    )


def test_entity_resolver_variant_match(resolver: EntityResolver) -> None:
    normalizer = ArabicNormalizer()
    result = resolver.resolve(
        "\u0627\u0644\u0625\u0645\u0627\u0645 \u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        "SCHOLAR",
    )
    assert result["match_type"] == "exact"
    assert result["canonical_name"] == normalizer.normalize(
        "\u0645\u062d\u0645\u062f \u0628\u0646 \u0625\u0633\u0645\u0627\u0639\u064a\u0644 \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    )


def test_entity_resolver_fuzzy_match(resolver: EntityResolver) -> None:
    result = resolver.resolve(
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a\u064a", "SCHOLAR"
    )
    assert result["match_type"] == "fuzzy"
    assert result["confidence"] >= 0.8


def test_entity_resolver_new_entity(resolver: EntityResolver) -> None:
    result = resolver.resolve(
        "\u0634\u062e\u0635 \u063a\u064a\u0631 \u0645\u0639\u0631\u0648\u0641",
        "SCHOLAR",
    )
    assert result["match_type"] == "new"
    assert result["confidence"] == 0.5


def test_entity_resolver_type_aware(resolver: EntityResolver) -> None:
    scholar = resolver.resolve("\u0645\u0633\u0644\u0645", "SCHOLAR")
    book = resolver.resolve("\u0645\u0633\u0644\u0645", "BOOK")
    assert scholar["canonical_name"] != book["canonical_name"]


def test_process_single_hadith_creates_nodes_and_edges(graph_builder) -> None:
    builder, _backend, _resolver = graph_builder
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0646\u0627\u0641\u0639",
        "\u0641\u064a",
        "\u0635\u062d\u064a\u062d",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        "\u062d\u062f\u064a\u062b",
        "\u0631\u0642\u0645",
        "1",
        "\u0627\u0644\u0631\u0628\u0627",
    ]
    labels = [
        "O",
        "B-SCHOLAR",
        "I-SCHOLAR",
        "O",
        "B-SCHOLAR",
        "O",
        "B-BOOK",
        "I-BOOK",
        "B-HADITH_REF",
        "I-HADITH_REF",
        "I-HADITH_REF",
        "B-CONCEPT",
    ]

    result = builder.process_hadith(tokens=tokens, labels=labels, hadith_id="h-single")
    stats = builder.get_stats()

    assert result["entities_inserted"] >= 5
    assert result["relations_inserted"] >= 3
    assert stats["nodes_by_label"].get("Scholar", 0) >= 2
    assert stats["relationships_by_type"].get("NARRATED_FROM", 0) == 1
    assert stats["relationships_by_type"].get("IN_BOOK", 0) == 1
    assert stats["relationships_by_type"].get("MENTIONS_CONCEPT", 0) == 1


def test_process_same_hadith_twice_has_no_duplicates(graph_builder) -> None:
    builder, _backend, _resolver = graph_builder
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0646\u0627\u0641\u0639",
    ]
    labels = ["O", "B-SCHOLAR", "I-SCHOLAR", "O", "B-SCHOLAR"]

    builder.process_hadith(tokens=tokens, labels=labels, hadith_id="h-dup")
    first_stats = builder.get_stats()
    builder.process_hadith(tokens=tokens, labels=labels, hadith_id="h-dup")
    second_stats = builder.get_stats()

    assert first_stats == second_stats


def test_entity_resolution_merges_variants_to_same_node(graph_builder) -> None:
    builder, backend, resolver = graph_builder
    normalizer = ArabicNormalizer()
    canonical = normalizer.normalize(
        "\u0645\u062d\u0645\u062f \u0628\u0646 \u0625\u0633\u0645\u0627\u0639\u064a\u0644 \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    )

    builder.process_hadith(
        tokens=["\u0642\u0627\u0644", "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"],
        labels=["O", "B-SCHOLAR"],
        hadith_id="h-var-1",
    )
    builder.process_hadith(
        tokens=[
            "\u0642\u0627\u0644",
            "\u0627\u0644\u0625\u0645\u0627\u0645",
            "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        ],
        labels=["O", "B-SCHOLAR", "I-SCHOLAR"],
        hadith_id="h-var-2",
    )

    scholars = backend.nodes.get("Scholar", {})
    assert canonical in scholars
    assert len([name for name in scholars if name == canonical]) == 1
    assert (
        resolver.resolve("\u0627\u0644\u0628\u062e\u0627\u0631\u064a", "SCHOLAR")[
            "canonical_name"
        ]
        == canonical
    )


def test_query_narration_chain_correct_order(graph_builder) -> None:
    builder, _backend, resolver = graph_builder
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0645\u0627\u0644\u0643",
        "\u0639\u0646",
        "\u0646\u0627\u0641\u0639",
    ]
    labels = ["O", "B-SCHOLAR", "I-SCHOLAR", "O", "B-SCHOLAR", "O", "B-SCHOLAR"]
    builder.process_hadith(tokens=tokens, labels=labels, hadith_id="h-chain")

    querier = GraphQuerier(builder.driver)
    chain = querier.get_narration_chain("h-chain")

    expected_s1 = resolver.resolve(
        "\u0639\u0628\u062f \u0627\u0644\u0644\u0647", "SCHOLAR"
    )["canonical_name"]
    expected_s2 = resolver.resolve("\u0645\u0627\u0644\u0643", "SCHOLAR")[
        "canonical_name"
    ]
    expected_s3 = resolver.resolve("\u0646\u0627\u0641\u0639", "SCHOLAR")[
        "canonical_name"
    ]

    assert len(chain) == 2
    assert chain[0]["source"] == expected_s1
    assert chain[0]["target"] == expected_s2
    assert chain[1]["source"] == expected_s2
    assert chain[1]["target"] == expected_s3


def test_graph_stats_counts_match_expected(graph_builder) -> None:
    builder, _backend, _resolver = graph_builder
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0646\u0627\u0641\u0639",
        "\u0641\u064a",
        "\u0635\u062d\u064a\u062d",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        "\u062d\u062f\u064a\u062b",
        "\u0631\u0642\u0645",
        "7",
        "\u0627\u0644\u0631\u0628\u0627",
    ]
    labels = [
        "O",
        "B-SCHOLAR",
        "I-SCHOLAR",
        "O",
        "B-SCHOLAR",
        "O",
        "B-BOOK",
        "I-BOOK",
        "B-HADITH_REF",
        "I-HADITH_REF",
        "I-HADITH_REF",
        "B-CONCEPT",
    ]
    builder.process_hadith(tokens=tokens, labels=labels, hadith_id="h-stats")
    stats = builder.get_stats()

    assert stats["relationships_by_type"].get("NARRATED_FROM", 0) == 1
    assert stats["relationships_by_type"].get("IN_BOOK", 0) == 1
    assert stats["relationships_by_type"].get("MENTIONS_CONCEPT", 0) == 1
    assert stats["nodes_by_label"].get("Scholar", 0) == 2
    assert stats["nodes_by_label"].get("Book", 0) == 1
    assert stats["nodes_by_label"].get("Concept", 0) == 1
    assert stats["nodes_by_label"].get("Hadith", 0) == 1
    assert stats["total_nodes"] == sum(stats["nodes_by_label"].values())
    assert stats["total_relationships"] == sum(stats["relationships_by_type"].values())
