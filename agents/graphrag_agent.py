from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Any

from database.graphmanager import GraphManager

try:
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:  # pragma: no cover - runtime dependency check
    ChatOpenAI = None
    OpenAIEmbeddings = None
    ChatPromptTemplate = None
    StrOutputParser = None
    load_dotenv = None


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_RESULT_LIMIT = 50
DEFAULT_SEED_LIMIT = 8
DEFAULT_NEIGHBORHOOD_LIMIT = 120
DEFAULT_NEIGHBORHOOD_HOPS = 2
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_MAX_RETRIEVAL_STEPS = 4
WRITE_KEYWORDS = {
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "DROP",
    "REMOVE",
    "SET",
    "LOAD",
    "CALL DBMS",
    "CALL APOC",
}


def quiet_terminal_noise() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*",
    )
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


@dataclass(frozen=True)
class RetrievalStep:
    kind: str
    query: str
    purpose: str


@dataclass(frozen=True)
class RetrievalResult:
    step: RetrievalStep
    seeds: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]
    cypher: str | None = None
    records: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class GraphRAGAnswer:
    question: str
    answer: str
    seeds: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]
    retrievals: list[RetrievalResult]
    cypher: str | None = None
    records: list[dict[str, Any]] | None = None
    retrieval_mode: str = "agentic_graph_rag"


def require_langchain() -> None:
    if (
        ChatOpenAI is None
        or OpenAIEmbeddings is None
        or ChatPromptTemplate is None
        or StrOutputParser is None
        or load_dotenv is None
    ):
        raise RuntimeError(
            "Missing LangChain dependencies. Install them with: "
            "py -m pip install -r requirements.txt"
        )


def make_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    require_langchain()
    load_dotenv(encoding="utf-8-sig")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(model=model, temperature=temperature)


def make_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    require_langchain()
    load_dotenv(encoding="utf-8-sig")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    return OpenAIEmbeddings(model=model)


def clean_cypher(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:cypher)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip().rstrip(";")


def validate_read_only_cypher(cypher: str) -> None:
    normalized = re.sub(r"\s+", " ", cypher.upper()).strip()
    if not normalized.startswith(("MATCH ", "OPTIONAL MATCH ", "WITH ", "UNWIND ")):
        raise ValueError("Generated Cypher must start with MATCH, OPTIONAL MATCH, WITH, or UNWIND.")

    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            raise ValueError(f"Generated Cypher contains a blocked keyword: {keyword}")


def ensure_limit(cypher: str, limit: int) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", cypher, flags=re.IGNORECASE):
        return cypher
    return f"{cypher}\nLIMIT {limit}"


def extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, list):
        raise ValueError("Retrieval plan must be a JSON array.")
    return [item for item in payload if isinstance(item, dict)]


class GraphRAGAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        result_limit: int = DEFAULT_RESULT_LIMIT,
        seed_limit: int = DEFAULT_SEED_LIMIT,
        neighborhood_hops: int = DEFAULT_NEIGHBORHOOD_HOPS,
        neighborhood_limit: int = DEFAULT_NEIGHBORHOOD_LIMIT,
        graph: GraphManager | None = None,
    ) -> None:
        self.llm = make_llm(model=model)
        self.embeddings = make_embeddings(model=embedding_model)
        self.result_limit = result_limit
        self.seed_limit = seed_limit
        self.neighborhood_hops = neighborhood_hops
        self.neighborhood_limit = neighborhood_limit
        self.graph = graph or GraphManager()
        self._schema_context: str | None = None

    def close(self) -> None:
        self.graph.close()

    def index_graph(
        self,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        limit: int | None = None,
    ) -> dict[str, int]:
        self.graph.create_vector_index()
        documents = self.graph.semantic_documents(limit=limit)

        indexed = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            texts = [document["search_text"] for document in batch]
            vectors = self.embeddings.embed_documents(texts)
            for document, vector in zip(batch, vectors, strict=True):
                self.graph.update_semantic_document(
                    node_id=document["id"],
                    search_text=document["search_text"],
                    embedding=vector,
                )
                indexed += 1

        return {"indexed_nodes": indexed, "total_documents": len(documents)}

    def schema_context(self) -> str:
        if self._schema_context is not None:
            return self._schema_context

        summary = self.graph.graph_summary()
        properties = self.graph.read(
            """
            MATCH (n:GraphEntity)
            UNWIND keys(n) AS property
            RETURN property, count(*) AS count
            ORDER BY count DESC, property
            LIMIT 80
            """
        )

        self._schema_context = json.dumps(
            {
                "labels": summary["labels"],
                "relationships": summary["relationships"],
                "common_properties": properties,
                "notes": [
                    "All imported nodes have the GraphEntity label and a stable id property.",
                    "Invoices connect to vendors with ISSUED_BY.",
                    "Vendors may connect to accounts with SERVES_ACCOUNT.",
                    "Accounts may connect to services with USES_SERVICE.",
                    "Services may connect to economics with HAS_CHARGE, HAS_USAGE, HAS_RATE, HAS_DISCOUNT, HAS_TAX_OR_FEE.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        return self._schema_context

    def semantic_retrieve(self, question: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        question_embedding = self.embeddings.embed_query(question)
        seeds = self.graph.semantic_search(question_embedding, top_k=self.seed_limit)
        seed_ids = [seed["id"] for seed in seeds if seed.get("id")]
        if not seed_ids:
            return seeds, []
        graph_context = self.graph.expand_neighborhood(
            node_ids=seed_ids,
            hops=self.neighborhood_hops,
            limit=self.neighborhood_limit,
        )
        return seeds, graph_context

    def plan_retrievals(self, question: str) -> list[RetrievalStep]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are planning retrieval steps for a Neo4j invoice GraphRAG system.

Return only a JSON array. Each item must have:
- kind: "semantic" or "cypher"
- query: natural-language semantic query or read-only Cypher
- purpose: why this retrieval is needed

Use multiple steps when the question needs several facts, such as identifying entities first and then connecting them to vendors/accounts/services/economics.
For semantic steps, write short focused search phrases like "regulatory surcharges taxes fees".
For cypher steps, use read-only Cypher only.
Never use write keywords.
Prefer 2-4 steps. Do not exceed {max_steps} steps.

Schema context:
{schema_context}
""",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        raw_plan = chain.invoke(
            {
                "question": question,
                "schema_context": self.schema_context(),
                "max_steps": DEFAULT_MAX_RETRIEVAL_STEPS,
            }
        )
        steps: list[RetrievalStep] = []
        for item in extract_json_array(raw_plan)[:DEFAULT_MAX_RETRIEVAL_STEPS]:
            kind = str(item.get("kind", "semantic")).strip().lower()
            query = str(item.get("query", "")).strip()
            purpose = str(item.get("purpose", "")).strip()
            if kind not in {"semantic", "cypher"} or not query:
                continue
            if kind == "cypher":
                query = ensure_limit(clean_cypher(query), self.result_limit)
                validate_read_only_cypher(query)
            steps.append(RetrievalStep(kind=kind, query=query, purpose=purpose))

        if not steps:
            steps.append(
                RetrievalStep(
                    kind="semantic",
                    query=question,
                    purpose="Find semantically relevant graph entities for the user question.",
                )
            )
        return steps

    def execute_retrieval_step(self, step: RetrievalStep) -> RetrievalResult:
        if step.kind == "cypher":
            records = self.graph.read(step.query)
            followup_text = f"{step.purpose}\n{step.query}\n{json.dumps(records[:10], default=str)}"
            seeds, graph_context = self.semantic_retrieve(followup_text)
            return RetrievalResult(
                step=step,
                seeds=seeds,
                graph_context=graph_context,
                cypher=step.query,
                records=records,
            )

        seeds, graph_context = self.semantic_retrieve(step.query)
        return RetrievalResult(step=step, seeds=seeds, graph_context=graph_context)

    def execute_retrieval_plan(self, question: str) -> list[RetrievalResult]:
        retrievals: list[RetrievalResult] = []
        for step in self.plan_retrievals(question):
            try:
                retrievals.append(self.execute_retrieval_step(step))
            except Exception:
                if step.kind == "cypher":
                    fallback = RetrievalStep(
                        kind="semantic",
                        query=f"{step.purpose} {question}",
                        purpose=f"Fallback semantic retrieval for failed Cypher step: {step.purpose}",
                    )
                    retrievals.append(self.execute_retrieval_step(fallback))
                else:
                    raise
        return retrievals

    def context_json(self, seeds: list[dict[str, Any]], graph_context: list[dict[str, Any]]) -> str:
        nodes: dict[str, dict[str, Any]] = {}
        relationships: dict[str, dict[str, Any]] = {}
        for path in graph_context:
            for node in path.get("nodes") or []:
                node_id = node.get("id")
                if node_id:
                    properties = {
                        key: value
                        for key, value in (node.get("properties") or {}).items()
                        if key != "embedding"
                    }
                    nodes[node_id] = {
                        "id": node_id,
                        "labels": node.get("labels"),
                        "properties": properties,
                    }
            for relationship in path.get("relationships") or []:
                relationship_id = relationship.get("id") or json.dumps(relationship, sort_keys=True)
                relationships[relationship_id] = relationship

        compact_seeds = [
            {
                "id": seed.get("id"),
                "labels": seed.get("labels"),
                "score": seed.get("score"),
                "properties": {
                    key: value
                    for key, value in (seed.get("properties") or {}).items()
                    if key != "embedding"
                },
                "search_text": seed.get("search_text"),
            }
            for seed in seeds
        ]

        return json.dumps(
            {
                "semantic_seed_nodes": compact_seeds,
                "expanded_nodes": list(nodes.values()),
                "expanded_relationships": list(relationships.values()),
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    def retrievals_context_json(self, retrievals: list[RetrievalResult]) -> str:
        all_nodes: dict[str, dict[str, Any]] = {}
        all_relationships: dict[str, dict[str, Any]] = {}
        retrieval_summaries: list[dict[str, Any]] = []

        for index, retrieval in enumerate(retrievals, start=1):
            seed_ids = [seed.get("id") for seed in retrieval.seeds if seed.get("id")]
            retrieval_summaries.append(
                {
                    "step": index,
                    "kind": retrieval.step.kind,
                    "purpose": retrieval.step.purpose,
                    "query": retrieval.step.query,
                    "seed_ids": seed_ids,
                    "cypher": retrieval.cypher,
                    "records": retrieval.records or [],
                }
            )
            for path in retrieval.graph_context:
                for node in path.get("nodes") or []:
                    node_id = node.get("id")
                    if not node_id:
                        continue
                    all_nodes[node_id] = {
                        "id": node_id,
                        "labels": node.get("labels"),
                        "properties": {
                            key: value
                            for key, value in (node.get("properties") or {}).items()
                            if key != "embedding"
                        },
                    }
                for relationship in path.get("relationships") or []:
                    relationship_id = relationship.get("id") or json.dumps(relationship, sort_keys=True)
                    all_relationships[relationship_id] = relationship

        return json.dumps(
            {
                "retrieval_steps": retrieval_summaries,
                "merged_nodes": list(all_nodes.values()),
                "merged_relationships": list(all_relationships.values()),
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    def generate_cypher(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You generate read-only Neo4j Cypher for an invoice graph database.

Rules:
- Return only Cypher. No Markdown.
- Use only read-only clauses: MATCH, OPTIONAL MATCH, WITH, WHERE, RETURN, ORDER BY, LIMIT, UNWIND.
- Never use CREATE, MERGE, SET, DELETE, REMOVE, DROP, LOAD, CALL, or APOC.
- Prefer GraphEntity paths and known labels/relationship types from the schema context.
- Include useful properties in RETURN, not just nodes, unless the user asks to view a graph.
- If the user asks to view a graph, return paths as p.
- Keep results focused and include LIMIT unless aggregating.

Schema context:
{schema_context}
""",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        cypher = clean_cypher(chain.invoke({"schema_context": self.schema_context(), "question": question}))
        validate_read_only_cypher(cypher)
        return ensure_limit(cypher, self.result_limit)

    def try_generate_cypher(self, question: str) -> tuple[str | None, list[dict[str, Any]]]:
        try:
            cypher = self.generate_cypher(question)
            return cypher, self.graph.read(cypher)
        except Exception:
            return None, []

    def answer_question(
        self,
        question: str,
        seeds: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
        cypher: str | None = None,
        records: list[dict[str, Any]] | None = None,
    ) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a GraphRAG assistant for invoice data stored in Neo4j.

Use the semantic seed nodes, expanded graph neighborhood, and optional Cypher records as your only factual sources.
Prefer explicit graph facts over inference. If the retrieved graph context does not contain enough evidence, say what is missing.
Be concise and include vendors, accounts, invoices, services, charges, fees, amounts, dates, and evidence when relevant.
""",
                ),
                (
                    "human",
                    """Question:
{question}

Semantic graph context:
{graph_context_json}

Optional Cypher:
{cypher}

Optional Cypher records:
{records_json}
""",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "question": question,
                "graph_context_json": self.context_json(seeds, graph_context),
                "cypher": cypher or "None",
                "records_json": json.dumps(records or [], ensure_ascii=False, default=str, indent=2),
            }
        ).strip()

    def answer_from_retrievals(self, question: str, retrievals: list[RetrievalResult]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an agentic GraphRAG assistant for invoice data in Neo4j.

You have performed one or more retrieval steps. Use only the provided retrieval context.
Reason across the retrieved graph facts before answering. Put together a polished, concise answer.

Answer style:
- Start with the direct answer in plain language.
- For simple questions, answer in 1-3 short sentences. Do not use a table for simple counts or lists.
- Use bullets only when the answer has several distinct items.
- Use a compact table only when comparing many vendors/accounts/services/fees and it improves readability.
- Include amounts, currencies, accounts, invoice names, and dates only when they directly answer the question.
- Include evidence snippets only when the user asks for evidence/source/why, or when the answer would otherwise be ambiguous.
- If the graph context is incomplete, mention the gap briefly at the end.
- Do not mention internal retrieval mechanics unless the user asks.
""",
                ),
                (
                    "human",
                    """Question:
{question}

Retrieval context:
{retrieval_context_json}
""",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "question": question,
                "retrieval_context_json": self.retrievals_context_json(retrievals),
            }
        ).strip()

    def ask(self, question: str) -> GraphRAGAnswer:
        retrievals = self.execute_retrieval_plan(question)
        seeds = [seed for retrieval in retrievals for seed in retrieval.seeds]
        graph_context = [path for retrieval in retrievals for path in retrieval.graph_context]
        cypher = next((retrieval.cypher for retrieval in retrievals if retrieval.cypher), None)
        records = [record for retrieval in retrievals for record in (retrieval.records or [])]
        answer = self.answer_from_retrievals(question, retrievals)
        return GraphRAGAnswer(
            question=question,
            answer=answer,
            seeds=seeds,
            graph_context=graph_context,
            retrievals=retrievals,
            cypher=cypher,
            records=records,
        )

    def __enter__(self) -> GraphRAGAgent:
        self.graph.verify_connectivity()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic GraphRAG over the Neo4j invoice graph.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model. Defaults to: {DEFAULT_MODEL}")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model. Defaults to: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_RESULT_LIMIT, help="Default Cypher LIMIT.")
    parser.add_argument("--show-cypher", action="store_true", help="Print the generated Cypher.")
    parser.add_argument("--show-seeds", action="store_true", help="Print semantic seed nodes.")

    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Create/update graph embeddings in Neo4j.")
    index_parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    index_parser.add_argument("--limit", type=int, default=None, help="Index only the first N graph nodes.")

    ask_parser = subparsers.add_parser("ask", help="Ask one question.")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--seeds", type=int, default=DEFAULT_SEED_LIMIT)
    ask_parser.add_argument("--hops", type=int, default=DEFAULT_NEIGHBORHOOD_HOPS)
    ask_parser.add_argument("--neighborhood-limit", type=int, default=DEFAULT_NEIGHBORHOOD_LIMIT)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chatbot.")
    chat_parser.add_argument("--seeds", type=int, default=DEFAULT_SEED_LIMIT)
    chat_parser.add_argument("--hops", type=int, default=DEFAULT_NEIGHBORHOOD_HOPS)
    chat_parser.add_argument("--neighborhood-limit", type=int, default=DEFAULT_NEIGHBORHOOD_LIMIT)
    return parser.parse_args()


def print_answer(
    result: GraphRAGAnswer,
    show_cypher: bool = False,
    show_seeds: bool = False,
) -> None:
    if show_seeds:
        print("\nRetrieval Plan:")
        for index, retrieval in enumerate(result.retrievals, start=1):
            print(
                f"{index}. {retrieval.step.kind}: {retrieval.step.purpose} "
                f"({retrieval.step.query})"
            )
    if show_seeds:
        print("\nSemantic Seeds:")
        for seed in result.seeds:
            props = seed.get("properties") or {}
            label_text = ", ".join(label for label in seed.get("labels", []) if label != "GraphEntity")
            name = (
                props.get("vendor_name")
                or props.get("customer_name")
                or props.get("description")
                or props.get("invoice_number")
                or props.get("id")
            )
            print(f"- {seed.get('id')} [{label_text}] score={seed.get('score'):.4f} {name}")
    if show_cypher:
        print("\nCypher:")
        print(result.cypher or "None")
    print("\nAnswer:")
    print(result.answer)


def chat(agent: GraphRAGAgent, show_cypher: bool = False, show_seeds: bool = False) -> None:
    print("GraphRAG chat ready. Type 'exit' or 'quit' to stop.")
    while True:
        question = input("\nQuestion> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return
        try:
            print_answer(agent.ask(question), show_cypher=show_cypher, show_seeds=show_seeds)
        except Exception as exc:
            print(f"GraphRAG error: {exc}", file=sys.stderr)


def main() -> int:
    quiet_terminal_noise()
    args = parse_args()
    command = args.command or "chat"

    try:
        with GraphRAGAgent(
            model=args.model,
            embedding_model=args.embedding_model,
            result_limit=args.limit,
            seed_limit=getattr(args, "seeds", DEFAULT_SEED_LIMIT),
            neighborhood_hops=getattr(args, "hops", DEFAULT_NEIGHBORHOOD_HOPS),
            neighborhood_limit=getattr(args, "neighborhood_limit", DEFAULT_NEIGHBORHOOD_LIMIT),
        ) as agent:
            if command == "index":
                counts = agent.index_graph(
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
                print(
                    "Indexed graph embeddings: "
                    f"{counts['indexed_nodes']}/{counts['total_documents']} nodes"
                )
            elif command == "ask":
                question = " ".join(args.question).strip()
                print_answer(
                    agent.ask(question),
                    show_cypher=args.show_cypher,
                    show_seeds=args.show_seeds,
                )
            else:
                chat(agent, show_cypher=args.show_cypher, show_seeds=args.show_seeds)
    except Exception as exc:
        print(f"GraphRAG failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
