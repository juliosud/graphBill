from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.invoice.invoice_extraction_orchestrator import InvoiceExtractionOrchestrator
from invoice_ingestor.config import IngestionConfig


@dataclass(frozen=True)
class IngestionResult:
    extraction_dir: Path
    reconstructed_invoice: Path
    graph_json: Path | None
    cleanup_completed: bool
    final_state: str


def ingest_invoice(config: IngestionConfig) -> IngestionResult:
    result = InvoiceExtractionOrchestrator(config).run()

    return IngestionResult(
        extraction_dir=result.extraction_dir,
        reconstructed_invoice=result.reconstructed_invoice,
        graph_json=result.graph_json,
        cleanup_completed=result.cleanup_completed,
        final_state=result.final_state.value,
    )
