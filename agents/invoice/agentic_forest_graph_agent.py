from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.invoice.invoice_graph_extraction_agent import (
    DEFAULT_MODEL,
    GRAPH_SCHEMA_VERSION,
    build_tools,
    extract_json_payload,
    get_agent_output,
    graph_extraction_user_prompt,
    make_llm,
    normalize_graph_payload,
)

try:
    from langchain.agents import create_agent
except ImportError:  # pragma: no cover - runtime dependency check
    create_agent = None


DEFAULT_FOREST_DIRNAME = "forest"


@dataclass(frozen=True)
class ForestTreeSpec:
    name: str
    system_prompt: str
    output_filename: str


@dataclass(frozen=True)
class ForestExtractionArtifacts:
    final_graph_path: Path
    qa_report_path: Path
    disagreements_path: Path
    tree_outputs: list[Path]
    raw_outputs: list[Path]


def require_forest_agent() -> None:
    if create_agent is None:
        raise RuntimeError(
            "Missing LangChain agent dependencies. Install them with: "
            "py -m pip install -r requirements.txt"
        )


def forest_tree_specs() -> list[ForestTreeSpec]:
    return [
        ForestTreeSpec(
            name="strict_evidence",
            output_filename="tree_strict_evidence_graph.json",
            system_prompt=(
                "You are the Strict Evidence tree in an agentic forest for telecom invoice graph extraction.\n\n"
                "Your personality:\n"
                "- Conservative and literal.\n"
                "- Extract only facts that are explicitly supported by the invoice text.\n"
                "- Prefer omission over invention.\n"
                "- If a hierarchy edge is plausible but not well-supported, place the ambiguity in unresolved.\n\n"
                "Mission:\n"
                "1. Inspect the invoice structure.\n"
                "2. Produce graph JSON using the canonical invoice graph schema.\n"
                "3. Mark every uncertainty in unresolved instead of guessing.\n\n"
                "Rules:\n"
                "- Return only valid JSON.\n"
                "- Use the shared canonical topology and allowed entity/relationship vocabulary.\n"
                "- Prefer explicit source text over normalization.\n"
                "- Preserve page evidence whenever possible.\n"
                "- Do not create computed totals unless the invoice clearly prints them.\n"
                "- When a field is weakly implied, leave it absent and explain in unresolved.\n"
                "- Before the final JSON, use the available tools to inspect structure and schema.\n"
                "- Do not wrap JSON in Markdown.\n"
            ),
        ),
        ForestTreeSpec(
            name="structural_layout",
            output_filename="tree_structural_layout_graph.json",
            system_prompt=(
                "You are the Structural Layout tree in an agentic forest for telecom invoice graph extraction.\n\n"
                "Your personality:\n"
                "- Table-aware and section-aware.\n"
                "- Use page markers, headings, row groups, bill-to sections, and repeated table structures.\n"
                "- Map document structure into the canonical graph topology.\n\n"
                "Mission:\n"
                "1. Detect document sections, account boundaries, and table regions.\n"
                "2. Convert that structure into canonical graph JSON.\n"
                "3. Preserve row-level evidence and section context.\n\n"
                "Rules:\n"
                "- Return only valid JSON.\n"
                "- Use the shared canonical topology and allowed entity/relationship vocabulary.\n"
                "- Prefer structural signals when deciding account, service, summary, and rollup boundaries.\n"
                "- Keep invoice line items and repeated tables complete when visible.\n"
                "- Use unresolved for missing structure rather than inventing it.\n"
                "- Before the final JSON, use the available tools to inspect structure and schema.\n"
                "- Do not wrap JSON in Markdown.\n"
            ),
        ),
        ForestTreeSpec(
            name="semantic_billing",
            output_filename="tree_semantic_billing_graph.json",
            system_prompt=(
                "You are the Semantic Billing tree in an agentic forest for telecom invoice graph extraction.\n\n"
                "Your personality:\n"
                "- Analytics-minded and telecom-aware.\n"
                "- Normalize services, usage, rates, charges, discounts, taxes, and summaries into the canonical topology.\n"
                "- Seek high recall, but keep evidence for every important claim.\n\n"
                "Mission:\n"
                "1. Interpret telecom billing meaning from the invoice text.\n"
                "2. Convert the document into canonical graph JSON.\n"
                "3. Separate explicit facts from evidence-backed assumptions.\n\n"
                "Rules:\n"
                "- Return only valid JSON.\n"
                "- Use the shared canonical topology and allowed entity/relationship vocabulary.\n"
                "- Normalize billing semantics when clearly supported by the text.\n"
                "- Preserve original vendor language in properties or evidence.\n"
                "- Put uncertain semantics in unresolved instead of forcing them.\n"
                "- Before the final JSON, use the available tools to inspect structure and schema.\n"
                "- Do not wrap JSON in Markdown.\n"
            ),
        ),
    ]


def forest_arbiter_system_prompt() -> str:
    return (
        "You are the arbiter in an agentic forest for telecom invoice graph extraction.\n\n"
        "You receive:\n"
        "- The original reconstructed invoice text\n"
        "- Three candidate graph JSON outputs targeting the same canonical topology\n"
        "- A deterministic disagreement summary\n\n"
        "Your job is to reconcile them into one final canonical graph and one QA report.\n\n"
        "Output requirements:\n"
        "- Return only valid JSON.\n"
        "- Top-level keys must be: final_graph, qa_report.\n"
        "- final_graph must follow the shared invoice graph contract with keys:\n"
        "  schema_version, source, analysis, entities, relationships, unresolved.\n"
        "- qa_report must include:\n"
        "  forest_version, tree_count, candidate_counts, consensus_summary,\n"
        "  supported_findings, disputed_findings, unresolved_findings,\n"
        "  hallucination_risks, confidence_score, import_recommendation.\n\n"
        "Arbitration rules:\n"
        "- Keep the canonical topology fixed.\n"
        "- Prefer facts supported by multiple trees and strong evidence.\n"
        "- A single-tree fact may survive only when the evidence is explicit and compelling.\n"
        "- Remove invented totals, dates, customers, vendors, relationships, or rollups.\n"
        "- If two interpretations are both plausible, keep the safer one and record the ambiguity in unresolved and qa_report.\n"
        "- Preserve useful normalization only when it is grounded in the source text.\n"
        "- Be conservative with unsupported hierarchy edges.\n"
        "- Do not mention internal chain-of-thought.\n"
        "- Do not wrap JSON in Markdown.\n"
    )


def forest_arbiter_user_prompt(
    txt_path: Path,
    invoice_text: str,
    tree_payloads: list[dict[str, Any]],
    disagreements: dict[str, Any],
) -> str:
    candidate_payload = {
        "source": {
            "txt_path": str(txt_path),
            "document_name": txt_path.parent.name,
        },
        "trees": tree_payloads,
        "disagreements": disagreements,
    }
    return (
        f"Reconcile these forest candidate outputs into one final invoice graph.\n\n"
        f"Text path: {txt_path}\n"
        f"Document name: {txt_path.parent.name}\n\n"
        "Reconstructed invoice text:\n"
        f"```text\n{invoice_text}\n```\n\n"
        "Candidate trees and disagreement summary:\n"
        f"```json\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n```\n"
    )


def build_disagreement_report(tree_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    entity_presence: dict[str, list[str]] = {}
    relationship_presence: dict[str, list[str]] = {}
    entity_type_counts: dict[str, dict[str, int]] = {}
    relationship_type_counts: dict[str, dict[str, int]] = {}

    for tree in tree_payloads:
        tree_name = str(tree.get("tree_name", "unknown"))
        payload = tree.get("payload") or {}

        for entity in payload.get("entities", []):
            entity_id = str(entity.get("id", ""))
            entity_type = str(entity.get("type", "Unknown"))
            if entity_id:
                entity_presence.setdefault(entity_id, []).append(tree_name)
            entity_type_counts.setdefault(entity_type, {})[tree_name] = (
                entity_type_counts.setdefault(entity_type, {}).get(tree_name, 0) + 1
            )

        for relationship in payload.get("relationships", []):
            relationship_id = str(relationship.get("id", ""))
            relationship_type = str(relationship.get("type", "Unknown"))
            if relationship_id:
                relationship_presence.setdefault(relationship_id, []).append(tree_name)
            relationship_type_counts.setdefault(relationship_type, {})[tree_name] = (
                relationship_type_counts.setdefault(relationship_type, {}).get(tree_name, 0) + 1
            )

    disputed_entities = [
        {"id": entity_id, "trees": sorted(trees)}
        for entity_id, trees in sorted(entity_presence.items())
        if len(set(trees)) < len(tree_payloads)
    ]
    disputed_relationships = [
        {"id": relationship_id, "trees": sorted(trees)}
        for relationship_id, trees in sorted(relationship_presence.items())
        if len(set(trees)) < len(tree_payloads)
    ]

    return {
        "tree_count": len(tree_payloads),
        "entity_type_counts": entity_type_counts,
        "relationship_type_counts": relationship_type_counts,
        "disputed_entities": disputed_entities[:200],
        "disputed_relationships": disputed_relationships[:200],
    }


def normalize_arbiter_payload(payload: dict[str, Any], txt_path: Path) -> dict[str, Any]:
    if "final_graph" not in payload:
        raise ValueError("Arbiter output is missing final_graph.")
    if "qa_report" not in payload:
        raise ValueError("Arbiter output is missing qa_report.")

    final_graph = normalize_graph_payload(payload["final_graph"], txt_path)
    qa_report = payload["qa_report"]
    if not isinstance(qa_report, dict):
        raise ValueError("Arbiter qa_report must be a JSON object.")
    qa_report.setdefault("forest_version", GRAPH_SCHEMA_VERSION + "-forest-v1")
    qa_report.setdefault("tree_count", 3)
    qa_report.setdefault("confidence_score", 0.0)
    qa_report.setdefault("import_recommendation", "review")
    return {"final_graph": final_graph, "qa_report": qa_report}


def run_forest_tree(
    txt_path: Path,
    invoice_text: str,
    spec: ForestTreeSpec,
    model: str,
    temperature: float,
) -> tuple[dict[str, Any], str]:
    require_forest_agent()
    llm = make_llm(model=model, temperature=temperature)
    tools = build_tools(invoice_text)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=spec.system_prompt,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": graph_extraction_user_prompt(txt_path, invoice_text),
                }
            ]
        }
    )
    raw_output = get_agent_output(result)
    payload = normalize_graph_payload(extract_json_payload(raw_output), txt_path)
    return {
        "tree_name": spec.name,
        "output_filename": spec.output_filename,
        "payload": payload,
    }, raw_output


def run_forest_arbiter(
    txt_path: Path,
    invoice_text: str,
    tree_payloads: list[dict[str, Any]],
    disagreements: dict[str, Any],
    model: str,
    temperature: float,
) -> tuple[dict[str, Any], str]:
    require_forest_agent()
    llm = make_llm(model=model, temperature=temperature)
    agent = create_agent(
        model=llm,
        tools=build_tools(invoice_text),
        system_prompt=forest_arbiter_system_prompt(),
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": forest_arbiter_user_prompt(
                        txt_path=txt_path,
                        invoice_text=invoice_text,
                        tree_payloads=tree_payloads,
                        disagreements=disagreements,
                    ),
                }
            ]
        }
    )
    raw_output = get_agent_output(result)
    return normalize_arbiter_payload(
        extract_json_payload(raw_output),
        txt_path=txt_path,
    ), raw_output


def extract_invoice_graph_with_forest(
    txt_path: Path,
    output: str | Path = "invoice_graph.json",
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    forest_dirname: str = DEFAULT_FOREST_DIRNAME,
) -> ForestExtractionArtifacts:
    txt_path = txt_path.resolve()
    invoice_text = txt_path.read_text(encoding="utf-8")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = txt_path.parent / output_path

    forest_dir = txt_path.parent / forest_dirname
    forest_dir.mkdir(parents=True, exist_ok=True)

    tree_payloads: list[dict[str, Any]] = []
    tree_paths: list[Path] = []
    raw_output_paths: list[Path] = []
    for spec in forest_tree_specs():
        tree_result, raw_output = run_forest_tree(
            txt_path=txt_path,
            invoice_text=invoice_text,
            spec=spec,
            model=model,
            temperature=temperature,
        )
        tree_payloads.append(tree_result)
        raw_output_path = forest_dir / spec.output_filename.replace(".json", ".raw.txt")
        raw_output_path.write_text(raw_output + "\n", encoding="utf-8")
        raw_output_paths.append(raw_output_path)
        tree_path = forest_dir / spec.output_filename
        tree_path.write_text(
            json.dumps(tree_result["payload"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tree_paths.append(tree_path)

    disagreements = build_disagreement_report(tree_payloads)
    disagreements_path = forest_dir / "disagreements.json"
    disagreements_path.write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    arbiter_payload, arbiter_raw_output = run_forest_arbiter(
        txt_path=txt_path,
        invoice_text=invoice_text,
        tree_payloads=tree_payloads,
        disagreements=disagreements,
        model=model,
        temperature=temperature,
    )
    arbiter_raw_output_path = forest_dir / "arbiter.raw.txt"
    arbiter_raw_output_path.write_text(arbiter_raw_output + "\n", encoding="utf-8")
    raw_output_paths.append(arbiter_raw_output_path)

    final_graph_path = output_path
    final_graph_path.parent.mkdir(parents=True, exist_ok=True)
    final_graph_path.write_text(
        json.dumps(arbiter_payload["final_graph"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    qa_report_path = forest_dir / "qa_report.json"
    qa_report_path.write_text(
        json.dumps(arbiter_payload["qa_report"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return ForestExtractionArtifacts(
        final_graph_path=final_graph_path,
        qa_report_path=qa_report_path,
        disagreements_path=disagreements_path,
        tree_outputs=tree_paths,
        raw_outputs=raw_output_paths,
    )
