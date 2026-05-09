from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionConfig:
    pdf_path: Path
    output_root: Path = Path("documents/txt")
    model: str = "gpt-5.5"
    temperature: float = 0
    y_tolerance: float = 3.0
    layout_width: int = 120
    skip_final_agent: bool = False
    reconstructed_filename: str = "reconstructed_invoice.txt"
    extract_graph_json: bool = True
    graph_json_filename: str = "invoice_graph.json"
    cleanup_extraction: bool = True
