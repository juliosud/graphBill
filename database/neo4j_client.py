from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase
    from neo4j import Driver
except ImportError:  # pragma: no cover - runtime dependency check
    Driver = Any
    GraphDatabase = None


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        load_dotenv()

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        missing = [
            name
            for name, value in {
                "NEO4J_URI": uri,
                "NEO4J_USERNAME": username,
                "NEO4J_PASSWORD": password,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Neo4j environment variables: {', '.join(missing)}")

        return cls(
            uri=str(uri),
            username=str(username),
            password=str(password),
            database=database,
        )


class Neo4jClient:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        if GraphDatabase is None:
            raise RuntimeError("Missing neo4j dependency. Install it with: py -m pip install neo4j")

        self.config = config or Neo4jConfig.from_env()
        self.driver: Driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def execute_write(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(cypher, parameters or {}).data())
            )
        return result

    def execute_read(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.execute_read(
                lambda tx: list(tx.run(cypher, parameters or {}).data())
            )
        return result

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
