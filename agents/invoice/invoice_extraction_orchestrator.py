from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agents.invoice.invoice_graph_extraction_agent import extract_invoice_graph
from agents.invoice.reconstruct_invoice_agent import reconstruct_invoice
from invoice_ingestor.config import IngestionConfig
from invoice_ingestor.extraction import extract_pdf


class InvoiceExtractionState(str, Enum):
    READY = "ready"
    EXTRACTING = "extracting"
    RECONSTRUCTING = "reconstructing"
    EXTRACTING_GRAPH = "extracting_graph"
    CLEANING_UP = "cleaning_up"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class InvoiceExtractionResult:
    extraction_dir: Path
    reconstructed_invoice: Path
    graph_json: Path | None
    cleanup_completed: bool
    final_state: InvoiceExtractionState


class InvoiceExtractionOrchestrator:
    """Coordinates invoice extraction, reconstruction, and cleanup as a state machine."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.state = InvoiceExtractionState.READY

    def run(self) -> InvoiceExtractionResult:
        extraction_dir: Path | None = None
        reconstructed_invoice: Path | None = None
        graph_json: Path | None = None
        cleanup_completed = False

        try:
            self.state = InvoiceExtractionState.EXTRACTING
            extraction_dir = extract_pdf(self.config.pdf_path, self.config.output_root)

            self.state = InvoiceExtractionState.RECONSTRUCTING
            reconstructed_invoice = reconstruct_invoice(
                extraction_dir=extraction_dir,
                output=self.config.reconstructed_filename,
                model=self.config.model,
                temperature=self.config.temperature,
                y_tolerance=self.config.y_tolerance,
                layout_width=self.config.layout_width,
                skip_final_agent=self.config.skip_final_agent,
            )

            if self.config.extract_graph_json:
                self.state = InvoiceExtractionState.EXTRACTING_GRAPH
                graph_json = extract_invoice_graph(
                    txt_path=reconstructed_invoice,
                    output=self.config.graph_json_filename,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )

            if self.config.cleanup_extraction:
                self.state = InvoiceExtractionState.CLEANING_UP
                self.cleanup_extraction_files(
                    extraction_dir=extraction_dir,
                    keep_files=[reconstructed_invoice, graph_json],
                )
                cleanup_completed = True

            self.state = InvoiceExtractionState.COMPLETED
            return InvoiceExtractionResult(
                extraction_dir=extraction_dir,
                reconstructed_invoice=reconstructed_invoice,
                graph_json=graph_json,
                cleanup_completed=cleanup_completed,
                final_state=self.state,
            )
        except Exception:
            self.state = InvoiceExtractionState.FAILED
            raise

    @staticmethod
    def cleanup_extraction_files(extraction_dir: Path, keep_files: list[Path | None]) -> None:
        keep_paths = {path.resolve() for path in keep_files if path is not None}

        for path in extraction_dir.iterdir():
            if path.resolve() in keep_paths:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
