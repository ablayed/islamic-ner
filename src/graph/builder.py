"""Neo4j knowledge graph builder for Islamic NER + RE output."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.graph.entity_resolver import EntityResolver
from src.relations.extract import RelationExtractor


class KnowledgeGraphBuilder:
    """
    Builds a Neo4j knowledge graph from NER + RE output.
    """

    _NODE_SCHEMA = {
        "SCHOLAR": ("Scholar", "canonical_name"),
        "BOOK": ("Book", "canonical_name"),
        "CONCEPT": ("Concept", "term"),
        "PLACE": ("Place", "canonical_name"),
        "HADITH_REF": ("Hadith", "hadith_id"),
        "HADITH": ("Hadith", "hadith_id"),
    }

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        *,
        driver=None,
        resolver: EntityResolver | None = None,
        relation_extractor: RelationExtractor | None = None,
    ):
        if driver is None:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        else:
            self.driver = driver

        self.resolver = resolver or EntityResolver()
        self.relation_extractor = relation_extractor or RelationExtractor()

    def close(self):
        self.driver.close()

    def clear_graph(self):
        """Delete all nodes and relationships."""
        with self.driver.session() as session:
            if hasattr(session, "clear_graph"):
                session.clear_graph()
                return
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        """Create uniqueness constraints."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scholar) REQUIRE s.canonical_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Book) REQUIRE b.canonical_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.term IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Place) REQUIRE p.canonical_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hadith) REQUIRE h.hadith_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for constraint in constraints:
                if hasattr(session, "create_constraint"):
                    session.create_constraint(constraint)
                else:
                    session.run(constraint)

    def insert_entities(self, entities: List[Dict]):
        """
        Insert resolved entities as nodes via MERGE.
        """
        unique_entities = {}
        for entity in entities:
            node = self._to_node(entity)
            if node is None:
                continue
            key = (node["entity_type"], node["key_value"])
            unique_entities[key] = node

        with self.driver.session() as session:
            for node in unique_entities.values():
                self._merge_entity_node(session, node)

        return len(unique_entities)

    def insert_relations(self, relations: List[Dict]):
        """
        Insert relation edges via MERGE.
        """
        relation_count = 0
        dedup = set()
        with self.driver.session() as session:
            for relation in relations:
                relation_type = str(relation.get("type", "")).upper()
                source_node = self._to_node(relation.get("source", {}))
                target_node = self._to_node(relation.get("target", {}))
                if not relation_type or source_node is None or target_node is None:
                    continue

                rel_props = {
                    "confidence": float(relation.get("confidence", 0.0)),
                    "evidence": str(relation.get("evidence", "")),
                    "source_hadith": str(relation.get("source_hadith", "")),
                }
                dedupe_key = (
                    relation_type,
                    source_node["label"],
                    source_node["key_value"],
                    target_node["label"],
                    target_node["key_value"],
                    rel_props["source_hadith"],
                )
                if dedupe_key in dedup:
                    continue
                dedup.add(dedupe_key)

                self._merge_entity_node(session, source_node)
                self._merge_entity_node(session, target_node)
                self._merge_relation(session, relation_type, source_node, target_node, rel_props)
                relation_count += 1

        return relation_count

    def process_hadith(
        self,
        tokens: List[str],
        labels: List[str],
        hadith_id: str | None = None,
        metadata: Dict | None = None,
    ):
        """
        Full pipeline for one hadith: NER spans -> resolve -> relations -> graph.
        """
        metadata = dict(metadata or {})
        if hadith_id is not None:
            metadata["hadith_id"] = str(hadith_id)
        hadith_id_str = str(metadata.get("hadith_id", "") or "")

        raw_entities = self.relation_extractor._extract_entity_spans(tokens, labels)
        resolved_entities: List[Dict] = []
        for entity in raw_entities:
            entity_type = str(entity.get("type", "")).upper()
            if entity_type in {"HADITH_REF", "HADITH"}:
                canonical_hadith = hadith_id_str or str(entity.get("text", ""))
                resolved_entities.append(
                    {
                        "entity_type": "HADITH_REF",
                        "canonical_name": canonical_hadith,
                        "original_text": str(entity.get("text", canonical_hadith)),
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                        "confidence": 1.0 if hadith_id_str else 0.7,
                        "match_type": "exact" if hadith_id_str else "new",
                        "book_ref": str(metadata.get("book_ref", "")),
                        "chapter": str(metadata.get("chapter", "")),
                    }
                )
                continue

            resolved = self.resolver.resolve(entity["text"], entity["type"])
            resolved_entities.append(
                {
                    "entity_type": resolved["entity_type"],
                    "canonical_name": resolved["canonical_name"],
                    "original_text": resolved["original_text"],
                    "start": int(entity["start"]),
                    "end": int(entity["end"]),
                    "confidence": float(resolved["confidence"]),
                    "match_type": resolved["match_type"],
                }
            )

        if hadith_id_str:
            resolved_entities.append(
                {
                    "entity_type": "HADITH_REF",
                    "canonical_name": hadith_id_str,
                    "original_text": hadith_id_str,
                    "confidence": 1.0,
                    "match_type": "exact",
                    "book_ref": str(metadata.get("book_ref", "")),
                    "chapter": str(metadata.get("chapter", "")),
                }
            )

        raw_relations = self.relation_extractor.extract(tokens, labels, metadata=metadata)
        relations = [self._resolve_relation_endpoints(rel, metadata) for rel in raw_relations]

        inserted_entities = self.insert_entities(resolved_entities)
        inserted_relations = self.insert_relations(relations)

        return {
            "entities_inserted": inserted_entities,
            "relations_inserted": inserted_relations,
        }

    def process_batch(self, hadiths: List[Dict], batch_size: int = 100):
        """
        Process multiple hadith dictionaries in sequence.
        """
        total_entities = 0
        total_relations = 0

        for idx, hadith in enumerate(hadiths, start=1):
            tokens = list(hadith.get("tokens", []))
            labels = list(hadith.get("labels", hadith.get("ner_tags", [])))
            hadith_id = hadith.get("id")
            metadata = dict(hadith.get("metadata", {}))

            result = self.process_hadith(tokens=tokens, labels=labels, hadith_id=hadith_id, metadata=metadata)
            total_entities += int(result["entities_inserted"])
            total_relations += int(result["relations_inserted"])

            if batch_size > 0 and idx % batch_size == 0:
                print(
                    f"Processed {idx}/{len(hadiths)} hadiths "
                    f"(entities={total_entities}, relations={total_relations})"
                )

        return {
            "hadiths_processed": len(hadiths),
            "entities_inserted": total_entities,
            "relations_inserted": total_relations,
        }

    def get_stats(self) -> Dict:
        """
        Return graph counts by node label and relationship type.
        """
        with self.driver.session() as session:
            if hasattr(session, "get_stats"):
                return session.get_stats()

            try:
                result = session.run(
                    "CALL apoc.meta.stats() YIELD labels, relTypes RETURN labels, relTypes"
                )
                record = result.single()
                if record is not None:
                    labels = dict(record.get("labels", {}))
                    rel_types = dict(record.get("relTypes", {}))
                    return {
                        "nodes_by_label": labels,
                        "relationships_by_type": rel_types,
                        "total_nodes": int(sum(labels.values())),
                        "total_relationships": int(sum(rel_types.values())),
                    }
            except Exception:
                pass

            node_rows = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
            )
            rel_rows = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"
            )
            nodes_by_label = {row["label"]: int(row["count"]) for row in node_rows}
            relationships_by_type = {row["type"]: int(row["count"]) for row in rel_rows}

            return {
                "nodes_by_label": nodes_by_label,
                "relationships_by_type": relationships_by_type,
                "total_nodes": int(sum(nodes_by_label.values())),
                "total_relationships": int(sum(relationships_by_type.values())),
            }

    def _resolve_relation_endpoints(self, relation: Dict, metadata: Dict) -> Dict:
        relation = dict(relation)
        source = dict(relation.get("source", {}))
        target = dict(relation.get("target", {}))

        source = self._resolve_endpoint(source, metadata)
        target = self._resolve_endpoint(target, metadata)
        relation["source"] = source
        relation["target"] = target

        if not relation.get("source_hadith"):
            relation["source_hadith"] = str(metadata.get("hadith_id", "") or "")
        return relation

    def _resolve_endpoint(self, endpoint: Dict, metadata: Dict) -> Dict:
        endpoint_type = str(endpoint.get("type", "")).upper()

        if endpoint_type in {"SCHOLAR", "BOOK", "CONCEPT", "PLACE"}:
            resolved = self.resolver.resolve(str(endpoint.get("text", "")), endpoint_type)
            endpoint["entity_type"] = resolved["entity_type"]
            endpoint["canonical_name"] = resolved["canonical_name"]
            endpoint["original_text"] = resolved["original_text"]
            endpoint["confidence"] = float(resolved["confidence"])
            endpoint["match_type"] = resolved["match_type"]
            return endpoint

        if endpoint_type in {"HADITH_REF", "HADITH"}:
            hadith_id = str(metadata.get("hadith_id", "") or endpoint.get("text", ""))
            endpoint["type"] = "HADITH_REF"
            endpoint["entity_type"] = "HADITH_REF"
            endpoint["canonical_name"] = hadith_id
            endpoint["original_text"] = hadith_id
            endpoint["book_ref"] = str(metadata.get("book_ref", ""))
            endpoint["chapter"] = str(metadata.get("chapter", ""))
            return endpoint

        return endpoint

    def _to_node(self, entity: Dict) -> Dict | None:
        entity_type = str(
            entity.get("entity_type") or entity.get("type") or ""
        ).upper()
        schema = self._NODE_SCHEMA.get(entity_type)
        if schema is None:
            return None
        label, key_field = schema

        if entity_type in {"HADITH_REF", "HADITH"}:
            key_value = str(
                entity.get("canonical_name")
                or entity.get("hadith_id")
                or entity.get("text")
                or ""
            )
            properties = {
                "hadith_id": key_value,
                "book_ref": str(entity.get("book_ref", "")),
                "chapter": str(entity.get("chapter", "")),
            }
        elif entity_type == "SCHOLAR":
            key_value = str(entity.get("canonical_name") or entity.get("text") or "")
            properties = {
                "canonical_name": key_value,
                "name_ar": str(entity.get("original_text", entity.get("text", key_value))),
                "variants": [str(entity.get("original_text", entity.get("text", key_value)))],
                "confidence": float(entity.get("confidence", 1.0)),
            }
        elif entity_type == "BOOK":
            key_value = str(entity.get("canonical_name") or entity.get("text") or "")
            properties = {
                "canonical_name": key_value,
                "title_ar": str(entity.get("original_text", entity.get("text", key_value))),
                "author": str(entity.get("author", "")),
            }
        elif entity_type == "CONCEPT":
            key_value = str(entity.get("canonical_name") or entity.get("text") or "")
            properties = {
                "term": key_value,
                "term_ar": str(entity.get("original_text", entity.get("text", key_value))),
                "category": str(entity.get("category", "")),
            }
        else:  # PLACE
            key_value = str(entity.get("canonical_name") or entity.get("text") or "")
            properties = {
                "canonical_name": key_value,
                "name_ar": str(entity.get("original_text", entity.get("text", key_value))),
            }

        if not key_value:
            return None

        return {
            "entity_type": entity_type,
            "label": label,
            "key_field": key_field,
            "key_value": key_value,
            "properties": properties,
        }

    def _merge_entity_node(self, session, node: Dict) -> None:
        if hasattr(session, "merge_entity_node"):
            session.merge_entity_node(
                node["label"],
                node["key_field"],
                node["key_value"],
                node["properties"],
            )
            return

        query = (
            f"MERGE (n:{node['label']} {{{node['key_field']}: $key_value}}) "
            "SET n += $properties"
        )
        session.run(query, key_value=node["key_value"], properties=node["properties"])

    def _merge_relation(
        self,
        session,
        relation_type: str,
        source_node: Dict,
        target_node: Dict,
        properties: Dict,
    ) -> None:
        if hasattr(session, "merge_relation"):
            session.merge_relation(
                relation_type=relation_type,
                source_node=source_node,
                target_node=target_node,
                properties=properties,
            )
            return

        query = f"""
        MATCH (s:{source_node['label']} {{{source_node['key_field']}: $source_key}})
        MATCH (t:{target_node['label']} {{{target_node['key_field']}: $target_key}})
        MERGE (s)-[r:{relation_type} {{source_hadith: $source_hadith}}]->(t)
        ON CREATE SET r.confidence = $confidence, r.evidence = $evidence
        ON MATCH SET
            r.confidence = CASE WHEN r.confidence < $confidence THEN $confidence ELSE r.confidence END,
            r.evidence = CASE WHEN r.evidence IS NULL OR r.evidence = '' THEN $evidence ELSE r.evidence END
        """
        session.run(
            query,
            source_key=source_node["key_value"],
            target_key=target_node["key_value"],
            source_hadith=str(properties.get("source_hadith", "")),
            confidence=float(properties.get("confidence", 0.0)),
            evidence=str(properties.get("evidence", "")),
        )
