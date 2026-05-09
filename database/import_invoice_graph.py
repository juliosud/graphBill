from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from database.neo4j_client import Neo4jClient


TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCALAR_TYPES = (str, int, float, bool)


def cypher_token(value: str, fallback: str) -> str:
    token = value.strip() if value else fallback
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    if not token or token[0].isdigit():
        token = f"{fallback}_{token}"
    if not TOKEN_PATTERN.fullmatch(token):
        raise ValueError(f"Could not create a safe Cypher token from: {value!r}")
    return token


def cypher_labels(entity: dict[str, Any]) -> str:
    labels = ["GraphEntity"]
    labels.extend(str(label) for label in entity.get("labels", []) if label)
    if entity.get("type"):
        labels.append(str(entity["type"]))

    unique_labels = []
    for label in labels:
        token = cypher_token(label, "Entity")
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


def entity_parameters(entity: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    properties = neo4j_properties(entity.get("properties"))
    properties.update(
        {
            "id": entity["id"],
            "entity_type": entity.get("type"),
            "graph_labels": [str(label) for label in entity.get("labels", [])],
            "evidence_json": neo4j_property_value(entity.get("evidence", [])),
            "source_document": source.get("document_name"),
            "source_txt_path": source.get("txt_path"),
        }
    )
    return properties


def relationship_parameters(relationship: dict[str, Any]) -> dict[str, Any]:
    properties = neo4j_properties(relationship.get("properties"))
    properties.update(
        {
            "id": relationship["id"],
            "relationship_type": relationship["type"],
            "evidence_json": neo4j_property_value(relationship.get("evidence", [])),
        }
    )
    return properties


def create_constraints(client: Neo4jClient) -> None:
    client.execute_write(
        """
        CREATE CONSTRAINT graph_entity_id IF NOT EXISTS
        FOR (n:GraphEntity)
        REQUIRE n.id IS UNIQUE
        """
    )


def clear_imported_graph(client: Neo4jClient) -> None:
    client.execute_write("MATCH (n:GraphEntity) DETACH DELETE n")


def import_invoice_graph(json_path: Path, clear_existing: bool = False) -> dict[str, int]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entities = payload.get("entities", [])
    relationships = payload.get("relationships", [])
    source = payload.get("source", {})
    entity_ids = {entity["id"] for entity in entities}
    missing_relationships = [
        relationship
        for relationship in relationships
        if relationship["source_id"] not in entity_ids or relationship["target_id"] not in entity_ids
    ]
    if missing_relationships:
        missing_ids = sorted(
            {
                node_id
                for relationship in missing_relationships
                for node_id in (relationship["source_id"], relationship["target_id"])
                if node_id not in entity_ids
            }
        )
        raise ValueError(
            "Graph JSON contains relationships with missing endpoint entities: "
            + ", ".join(missing_ids)
        )

    with Neo4jClient() as client:
        client.verify_connectivity()
        create_constraints(client)

        if clear_existing:
            clear_imported_graph(client)

        for entity in entities:
            labels = cypher_labels(entity)
            client.execute_write(
                f"""
                MERGE (n:GraphEntity {{id: $id}})
                SET n += $properties
                SET n{labels}
                """,
                {"id": entity["id"], "properties": entity_parameters(entity, source)},
            )

        for relationship in relationships:
            relationship_type = cypher_token(relationship["type"], "RELATED_TO")
            client.execute_write(
                f"""
                MATCH (source:GraphEntity {{id: $source_id}})
                MATCH (target:GraphEntity {{id: $target_id}})
                MERGE (source)-[r:{relationship_type} {{id: $id}}]->(target)
                SET r += $properties
                """,
                {
                    "source_id": relationship["source_id"],
                    "target_id": relationship["target_id"],
                    "id": relationship["id"],
                    "properties": relationship_parameters(relationship),
                },
            )

    return {"nodes": len(entities), "relationships": len(relationships)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import invoice graph JSON into Neo4j.")
    parser.add_argument("json_path", type=Path, help="Path to invoice_graph.json")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing imported GraphEntity nodes before importing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = import_invoice_graph(args.json_path, clear_existing=args.clear_existing)
    print(f"Imported {counts['nodes']} nodes and {counts['relationships']} relationships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
