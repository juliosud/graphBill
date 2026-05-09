from __future__ import annotations

import json
import re
from typing import Any, Literal

from database.neo4j_client import Neo4jClient, Neo4jConfig


Direction = Literal["incoming", "outgoing", "both"]

TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCALAR_TYPES = (str, int, float, bool)
VECTOR_INDEX_NAME = "graph_entity_embedding"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def cypher_token(value: str, fallback: str) -> str:
    token = value.strip() if value else fallback
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    if not token or token[0].isdigit():
        token = f"{fallback}_{token}"
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError(f"Could not create a safe Cypher token from: {value!r}")
    return token


def cypher_labels(labels: list[str] | None) -> str:
    unique_labels = ["GraphEntity"]
    for label in labels or []:
        token = cypher_token(str(label), "Entity")
        if token not in unique_labels:
            unique_labels.append(token)
    return "".join(f":{label}" for label in unique_labels)


def neo4j_property_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, SCALAR_TYPES):
        return value
    if isinstance(value, list) and all(isinstance(item, SCALAR_TYPES) for item in value):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def neo4j_properties(raw: dict[str, Any] | None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if value is None:
            continue
        properties[str(key)] = neo4j_property_value(value)
    return properties


def compact_property_value(value: Any, max_length: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text


def graph_entity_search_text(
    labels: list[str],
    properties: dict[str, Any],
    neighborhood: list[dict[str, Any]] | None = None,
) -> str:
    ignored_properties = {"embedding", "search_text"}
    lines = [
        f"Labels: {', '.join(label for label in labels if label != 'GraphEntity')}",
        f"ID: {properties.get('id', '')}",
        f"Entity type: {properties.get('entity_type', '')}",
        f"Source document: {properties.get('source_document', '')}",
    ]

    for key, value in sorted(properties.items()):
        if key in ignored_properties or value in (None, "", []):
            continue
        lines.append(f"{key}: {compact_property_value(value)}")

    if neighborhood:
        lines.append("Connected graph facts:")
        for item in neighborhood[:20]:
            other = item.get("other_properties") or {}
            other_labels = ", ".join(item.get("other_labels") or [])
            other_name = (
                other.get("name")
                or other.get("vendor_name")
                or other.get("customer_name")
                or other.get("description")
                or other.get("id")
                or ""
            )
            lines.append(
                f"{item.get('direction')} {item.get('type')} "
                f"{other_labels} {compact_property_value(other_name, max_length=160)}"
            )

    return "\n".join(line for line in lines if line.strip())


class GraphManager:
    """High-level graph operations for imported invoice entities."""

    def __init__(self, config: Neo4jConfig | None = None, client: Neo4jClient | None = None) -> None:
        self.client = client or Neo4jClient(config=config)

    def close(self) -> None:
        self.client.close()

    def verify_connectivity(self) -> None:
        self.client.verify_connectivity()

    def create_constraints(self) -> None:
        self.client.execute_write(
            """
            CREATE CONSTRAINT graph_entity_id IF NOT EXISTS
            FOR (n:GraphEntity)
            REQUIRE n.id IS UNIQUE
            """
        )

    def create_vector_index(
        self,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        index_name: str = VECTOR_INDEX_NAME,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("Vector index dimensions must be positive.")

        self.client.execute_write(
            f"""
            CREATE VECTOR INDEX {cypher_token(index_name, "graph_entity_embedding")} IF NOT EXISTS
            FOR (n:GraphEntity)
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )

    def read(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.client.execute_read(cypher, parameters)

    def write(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.client.execute_write(cypher, parameters)

    def create_node(
        self,
        node_id: str,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        label_clause = cypher_labels(labels)
        rows = self.client.execute_write(
            f"""
            MERGE (n:GraphEntity {{id: $id}})
            SET n += $properties
            SET n{label_clause}
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            """,
            {"id": node_id, "properties": {"id": node_id, **neo4j_properties(properties)}},
        )
        return rows[0] if rows else None

    def update_node(self, node_id: str, properties: dict[str, Any]) -> dict[str, Any] | None:
        rows = self.client.execute_write(
            """
            MATCH (n:GraphEntity {id: $id})
            SET n += $properties
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            """,
            {"id": node_id, "properties": neo4j_properties(properties)},
        )
        return rows[0] if rows else None

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self.client.execute_read(
            """
            MATCH (n:GraphEntity {id: $id})
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            """,
            {"id": node_id},
        )
        return rows[0] if rows else None

    def find_nodes(
        self,
        label: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        label_clause = f":{cypher_token(label, 'Entity')}" if label else ""
        return self.client.execute_read(
            f"""
            MATCH (n:GraphEntity{label_clause})
            WHERE all(key IN keys($filters) WHERE n[key] = $filters[key])
            RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
            ORDER BY n.id
            LIMIT $limit
            """,
            {"filters": neo4j_properties(filters), "limit": limit},
        )

    def delete_node(self, node_id: str, detach: bool = True) -> bool:
        delete_clause = "DETACH DELETE n" if detach else "DELETE n"
        rows = self.client.execute_write(
            f"""
            MATCH (n:GraphEntity {{id: $id}})
            WITH n, n.id AS deleted_id
            {delete_clause}
            RETURN deleted_id
            """,
            {"id": node_id},
        )
        return bool(rows)

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        relationship_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        relationship_token = cypher_token(relationship_type, "RELATED_TO")
        rel_id = relationship_id or f"{source_id}_{relationship_token}_{target_id}"
        rel_properties = {
            "id": rel_id,
            "relationship_type": relationship_token,
            **neo4j_properties(properties),
        }
        rows = self.client.execute_write(
            f"""
            MATCH (source:GraphEntity {{id: $source_id}})
            MATCH (target:GraphEntity {{id: $target_id}})
            MERGE (source)-[r:{relationship_token} {{id: $id}}]->(target)
            SET r += $properties
            RETURN
                r.id AS id,
                type(r) AS type,
                source.id AS source_id,
                target.id AS target_id,
                properties(r) AS properties
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "id": rel_id,
                "properties": rel_properties,
            },
        )
        return rows[0] if rows else None

    def update_relationship(
        self,
        relationship_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any] | None:
        rows = self.client.execute_write(
            """
            MATCH (source:GraphEntity)-[r {id: $id}]->(target:GraphEntity)
            SET r += $properties
            RETURN
                r.id AS id,
                type(r) AS type,
                source.id AS source_id,
                target.id AS target_id,
                properties(r) AS properties
            """,
            {"id": relationship_id, "properties": neo4j_properties(properties)},
        )
        return rows[0] if rows else None

    def get_relationships(
        self,
        node_id: str | None = None,
        relationship_type: str | None = None,
        direction: Direction = "both",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        relationship_clause = ""
        if relationship_type:
            relationship_clause = f":{cypher_token(relationship_type, 'RELATED_TO')}"

        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be one of: incoming, outgoing, both")

        if node_id and direction == "outgoing":
            pattern = f"(source:GraphEntity {{id: $node_id}})-[r{relationship_clause}]->(target:GraphEntity)"
            where_clause = ""
        elif node_id and direction == "incoming":
            pattern = f"(source:GraphEntity)-[r{relationship_clause}]->(target:GraphEntity {{id: $node_id}})"
            where_clause = ""
        elif node_id:
            pattern = f"(source:GraphEntity)-[r{relationship_clause}]-(target:GraphEntity)"
            where_clause = "WHERE source.id = $node_id OR target.id = $node_id"
        else:
            pattern = f"(source:GraphEntity)-[r{relationship_clause}]->(target:GraphEntity)"
            where_clause = ""

        return self.client.execute_read(
            f"""
            MATCH {pattern}
            {where_clause}
            RETURN
                r.id AS id,
                type(r) AS type,
                source.id AS source_id,
                target.id AS target_id,
                properties(r) AS properties
            ORDER BY type, id
            LIMIT $limit
            """,
            {"node_id": node_id, "limit": limit},
        )

    def delete_relationship(self, relationship_id: str) -> bool:
        rows = self.client.execute_write(
            """
            MATCH ()-[r {id: $id}]->()
            WITH r, r.id AS deleted_id
            DELETE r
            RETURN deleted_id
            """,
            {"id": relationship_id},
        )
        return bool(rows)

    def clear_graph(self) -> None:
        self.client.execute_write("MATCH (n:GraphEntity) DETACH DELETE n")

    def semantic_documents(self, limit: int | None = None) -> list[dict[str, Any]]:
        limit_clause = "LIMIT $limit" if limit is not None else ""
        rows = self.client.execute_read(
            f"""
            MATCH (n:GraphEntity)
            OPTIONAL MATCH (n)-[r]-(m:GraphEntity)
            WITH n, collect({{
                type: type(r),
                direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
                other_id: m.id,
                other_labels: labels(m),
                other_properties: properties(m)
            }}) AS neighborhood
            RETURN
                n.id AS id,
                labels(n) AS labels,
                properties(n) AS properties,
                neighborhood AS neighborhood
            ORDER BY n.id
            {limit_clause}
            """,
            {"limit": limit},
        )
        return [
            {
                "id": row["id"],
                "search_text": graph_entity_search_text(
                    labels=row.get("labels") or [],
                    properties=row.get("properties") or {},
                    neighborhood=[
                        item
                        for item in row.get("neighborhood") or []
                        if item.get("type") and item.get("other_id")
                    ],
                ),
            }
            for row in rows
        ]

    def update_semantic_document(self, node_id: str, search_text: str, embedding: list[float]) -> None:
        self.client.execute_write(
            """
            MATCH (n:GraphEntity {id: $id})
            SET n.search_text = $search_text,
                n.embedding = $embedding
            """,
            {"id": node_id, "search_text": search_text, "embedding": embedding},
        )

    def semantic_search(
        self,
        embedding: list[float],
        top_k: int = 8,
        index_name: str = VECTOR_INDEX_NAME,
    ) -> list[dict[str, Any]]:
        return self.client.execute_read(
            f"""
            CALL db.index.vector.queryNodes(
                '{cypher_token(index_name, "graph_entity_embedding")}',
                $top_k,
                $embedding
            )
            YIELD node, score
            RETURN
                node.id AS id,
                labels(node) AS labels,
                properties(node) AS properties,
                node.search_text AS search_text,
                score AS score
            ORDER BY score DESC
            """,
            {"top_k": top_k, "embedding": embedding},
        )

    def expand_neighborhood(self, node_ids: list[str], hops: int = 2, limit: int = 120) -> list[dict[str, Any]]:
        return self.client.execute_read(
            """
            MATCH path = (seed:GraphEntity)-[*0..2]-(neighbor:GraphEntity)
            WHERE seed.id IN $node_ids
            WITH path, seed, neighbor
            LIMIT $limit
            RETURN
                seed.id AS seed_id,
                [node IN nodes(path) | {
                    id: node.id,
                    labels: labels(node),
                    properties: properties(node)
                }] AS nodes,
                [rel IN relationships(path) | {
                    id: rel.id,
                    type: type(rel),
                    source_id: startNode(rel).id,
                    target_id: endNode(rel).id,
                    properties: properties(rel)
                }] AS relationships
            """.replace("*0..2", f"*0..{max(1, min(hops, 4))}"),
            {"node_ids": node_ids, "limit": limit},
        )

    def graph_summary(self) -> dict[str, list[dict[str, Any]]]:
        labels = self.client.execute_read(
            """
            MATCH (n:GraphEntity)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC, label
            """
        )
        relationships = self.client.execute_read(
            """
            MATCH (:GraphEntity)-[r]->(:GraphEntity)
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC, type
            """
        )
        return {"labels": labels, "relationships": relationships}

    def __enter__(self) -> GraphManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
