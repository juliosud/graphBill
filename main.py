from __future__ import annotations

import sys
from pathlib import Path

from invoice_ingestor import IngestionConfig, ingest_invoice


# Put the invoice PDF in documents/pdf, then update only this path.
PDF_PATH = Path("documents/pdf/Telecom Europe.pdf")


OUTPUT_ROOT = Path("documents/txt")
MODEL = "gpt-5.5"
TEMPERATURE = 0
Y_TOLERANCE = 3.0
LAYOUT_WIDTH = 120
SKIP_FINAL_AGENT = False
EXTRACT_GRAPH_JSON = True
GRAPH_JSON_FILENAME = "invoice_graph.json"
CLEANUP_EXTRACTION = True


def main() -> int:
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}", file=sys.stderr)
        print("Place the invoice PDF in documents/pdf or update PDF_PATH in main.py.", file=sys.stderr)
        return 1

    try:
        result = ingest_invoice(
            IngestionConfig(
                pdf_path=PDF_PATH,
                output_root=OUTPUT_ROOT,
                model=MODEL,
                temperature=TEMPERATURE,
                y_tolerance=Y_TOLERANCE,
                layout_width=LAYOUT_WIDTH,
                skip_final_agent=SKIP_FINAL_AGENT,
                extract_graph_json=EXTRACT_GRAPH_JSON,
                graph_json_filename=GRAPH_JSON_FILENAME,
                cleanup_extraction=CLEANUP_EXTRACTION,
            )
        )
    except Exception as exc:
        print(f"Invoice ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Extraction folder: {result.extraction_dir}")
    print(f"Reconstructed invoice: {result.reconstructed_invoice}")
    print(f"Invoice graph JSON: {result.graph_json}")
    print(f"Cleaned intermediate files: {result.cleanup_completed}")
    print(f"Final state: {result.final_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
