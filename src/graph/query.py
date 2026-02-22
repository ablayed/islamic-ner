"""Common graph query helpers for the Islamic knowledge graph."""

from __future__ import annotations

from typing import Dict, List


class GraphQuerier:
    """Common queries for the Islamic knowledge graph."""

    def __init__(self, driver):
        self.driver = driver

    def find_scholar(self, name: str) -> Dict:
        """Find a scholar node by (fuzzy) name lookup."""
        with self.driver.session() as session:
            if hasattr(session, "find_scholar"):
                return session.find_scholar(name)

            result = session.run(
                """
                MATCH (s:Scholar)
                WHERE s.canonical_name CONTAINS $name
                   OR coalesce(s.name_ar, '') CONTAINS $name
                RETURN s.canonical_name AS canonical_name,
                       coalesce(s.name_ar, s.canonical_name) AS name_ar
                ORDER BY size(s.canonical_name)
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            if record is None:
                return {}
            return {
                "canonical_name": record["canonical_name"],
                "name_ar": record["name_ar"],
            }

    def get_narration_chain(self, hadith_id: str) -> List[Dict]:
        """Get the narration chain edges for a specific hadith id."""
        with self.driver.session() as session:
            if hasattr(session, "get_narration_chain"):
                return session.get_narration_chain(hadith_id)

            result = session.run(
                """
                MATCH (s:Scholar)-[r:NARRATED_FROM]->(t:Scholar)
                WHERE r.source_hadith = $hadith_id
                RETURN s.canonical_name AS source,
                       t.canonical_name AS target,
                       r.confidence AS confidence,
                       r.evidence AS evidence
                ORDER BY source, target
                """,
                hadith_id=hadith_id,
            )
            return [
                {
                    "source": row["source"],
                    "target": row["target"],
                    "confidence": float(row["confidence"]),
                    "evidence": row["evidence"],
                }
                for row in result
            ]

    def get_scholar_connections(self, scholar_name: str) -> Dict:
        """Get teachers and students connected by narration edges."""
        with self.driver.session() as session:
            if hasattr(session, "get_scholar_connections"):
                return session.get_scholar_connections(scholar_name)

            teachers = session.run(
                """
                MATCH (s:Scholar {canonical_name: $name})-[:NARRATED_FROM]->(teacher:Scholar)
                RETURN DISTINCT teacher.canonical_name AS name
                ORDER BY name
                """,
                name=scholar_name,
            )
            students = session.run(
                """
                MATCH (student:Scholar)-[:NARRATED_FROM]->(s:Scholar {canonical_name: $name})
                RETURN DISTINCT student.canonical_name AS name
                ORDER BY name
                """,
                name=scholar_name,
            )
            return {
                "scholar": scholar_name,
                "teachers": [row["name"] for row in teachers],
                "students": [row["name"] for row in students],
            }

    def get_concepts_in_book(self, book_name: str) -> List[str]:
        """Get all concepts linked to hadiths in a given book."""
        with self.driver.session() as session:
            if hasattr(session, "get_concepts_in_book"):
                return session.get_concepts_in_book(book_name)

            result = session.run(
                """
                MATCH (h:Hadith)-[:IN_BOOK]->(b:Book)
                MATCH (h)-[:MENTIONS_CONCEPT]->(c:Concept)
                WHERE b.canonical_name CONTAINS $book_name
                   OR coalesce(b.title_ar, '') CONTAINS $book_name
                RETURN DISTINCT c.term AS concept
                ORDER BY concept
                """,
                book_name=book_name,
            )
            return [row["concept"] for row in result]

    def shortest_path(self, scholar1: str, scholar2: str) -> List[str]:
        """Find shortest undirected narration path between two scholars."""
        with self.driver.session() as session:
            if hasattr(session, "shortest_path"):
                return session.shortest_path(scholar1, scholar2)

            result = session.run(
                """
                MATCH (a:Scholar {canonical_name: $scholar1}),
                      (b:Scholar {canonical_name: $scholar2})
                MATCH p = shortestPath((a)-[:NARRATED_FROM*..20]-(b))
                RETURN [node IN nodes(p) | node.canonical_name] AS path
                LIMIT 1
                """,
                scholar1=scholar1,
                scholar2=scholar2,
            )
            record = result.single()
            if record is None:
                return []
            return list(record["path"])
