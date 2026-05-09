from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from agents.graphrag_agent import GraphRAGAgent
from database.import_invoice_graph import import_invoice_graph
from invoice_ingestor import IngestionConfig, ingest_invoice


OUTPUT_ROOT = Path("documents/txt")
MODEL = "gpt-5.5"
TEMPERATURE = 0
Y_TOLERANCE = 3.0
LAYOUT_WIDTH = 120
GRAPH_JSON_FILENAME = "invoice_graph.json"


def log(message: str) -> None:
    print(message, flush=True)


def log_stage(stage: str, message: str) -> None:
    log(f"[{stage}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one invoice PDF, push its graph to Neo4j, and refresh GraphRAG embeddings."
    )
    parser.add_argument("pdf", type=Path, help="Path to one invoice PDF.")
    parser.add_argument(
        "--clear-graph",
        action="store_true",
        help="Clear existing imported GraphEntity data before importing this invoice.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip rebuilding the semantic GraphRAG index after import.",
    )
    parser.add_argument(
        "--skip-final-agent",
        action="store_true",
        help="Skip final invoice reconstruction agent and use extracted layout text directly.",
    )
    return parser.parse_args()


def run_invoice_pipeline(args: argparse.Namespace) -> None:
    pipeline_started_at = perf_counter()
    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log_stage("start", f"Running invoice pipeline for: {pdf_path}")
    log_stage("config", f"Output root: {OUTPUT_ROOT}")
    log_stage("config", f"Model: {MODEL}")
    log_stage("extract", "Starting PDF extraction, invoice reconstruction, and graph JSON extraction.")
    extraction_started_at = perf_counter()
    result = ingest_invoice(
        IngestionConfig(
            pdf_path=pdf_path,
            output_root=OUTPUT_ROOT,
            model=MODEL,
            temperature=TEMPERATURE,
            y_tolerance=Y_TOLERANCE,
            layout_width=LAYOUT_WIDTH,
            skip_final_agent=args.skip_final_agent,
            extract_graph_json=True,
            graph_json_filename=GRAPH_JSON_FILENAME,
            cleanup_extraction=True,
        )
    )
    log_stage("extract", f"Completed in {perf_counter() - extraction_started_at:.1f}s")

    if result.graph_json is None:
        raise RuntimeError("Invoice extraction completed without graph JSON.")

    log_stage("extract", f"Extraction folder: {result.extraction_dir}")
    log_stage("extract", f"Reconstructed invoice: {result.reconstructed_invoice}")
    log_stage("extract", f"Invoice graph JSON: {result.graph_json}")
    log_stage("extract", f"Cleaned intermediate files: {result.cleanup_completed}")
    log_stage("extract", f"Final state: {result.final_state}")

    log_stage("neo4j", "Importing invoice graph into Neo4j.")
    if args.clear_graph:
        log_stage("neo4j", "Clearing existing imported graph data before import.")
    import_started_at = perf_counter()
    counts = import_invoice_graph(result.graph_json, clear_existing=args.clear_graph)
    log_stage(
        "neo4j",
        "Import complete: "
        f"{counts['nodes']} nodes, {counts['relationships']} relationships",
    )
    log_stage("neo4j", f"Completed in {perf_counter() - import_started_at:.1f}s")

    if args.skip_index:
        log_stage("index", "Skipped GraphRAG semantic index refresh.")
        log_stage("done", f"Pipeline completed in {perf_counter() - pipeline_started_at:.1f}s")
        return

    log_stage("index", "Refreshing GraphRAG semantic index.")
    index_started_at = perf_counter()
    with GraphRAGAgent() as agent:
        index_counts = agent.index_graph()
    log_stage(
        "index",
        "Refresh complete: "
        f"{index_counts['indexed_nodes']}/{index_counts['total_documents']} nodes",
    )
    log_stage("index", f"Completed in {perf_counter() - index_started_at:.1f}s")
    log_stage("done", f"Pipeline completed in {perf_counter() - pipeline_started_at:.1f}s")


def main() -> int:
    args = parse_args()
    try:
        run_invoice_pipeline(args)
    except Exception as exc:
        print(f"Invoice pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
