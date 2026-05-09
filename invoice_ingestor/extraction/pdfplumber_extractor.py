from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    import pdfplumber
except ImportError:  # pragma: no cover - helpful runtime message
    pdfplumber = None


def make_json_safe(value: Any) -> Any:
    """Convert pdfplumber/PDFMiner values into JSON-serializable data."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_table_csv(path: Path, table: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(table)


def extract_pdf(pdf_path: Path, output_root: Path) -> Path:
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is not installed. Install it with: python -m pip install pdfplumber"
        )

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extraction_dir = output_root / pdf_path.stem
    pages_dir = extraction_dir / "pages"
    tables_dir = extraction_dir / "tables"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    document_text: list[str] = []
    summary: dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "output_folder": str(extraction_dir.resolve()),
        "pages": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        summary["metadata"] = pdf.metadata
        summary["page_count"] = len(pdf.pages)
        write_json(extraction_dir / "metadata.json", pdf.metadata)

        for page in pdf.pages:
            page_number = page.page_number
            prefix = f"page_{page_number:04d}"

            text = page.extract_text() or ""
            words = page.extract_words(
                keep_blank_chars=True,
                use_text_flow=True,
                extra_attrs=["fontname", "size"],
            )
            tables = page.extract_tables()

            page_payload = {
                "page_number": page_number,
                "width": page.width,
                "height": page.height,
                "rotation": page.rotation,
                "bbox": page.bbox,
                "cropbox": getattr(page, "cropbox", None),
                "mediabox": getattr(page, "mediabox", None),
                "text": text,
                "words": words,
                "chars": page.chars,
                "lines": page.lines,
                "rects": page.rects,
                "curves": page.curves,
                "images": page.images,
                "annots": getattr(page, "annots", []),
                "hyperlinks": getattr(page, "hyperlinks", []),
                "objects": page.objects,
                "tables": tables,
            }

            write_json(pages_dir / f"{prefix}.json", page_payload)

            if text:
                document_text.append(f"\n\n--- Page {page_number} ---\n\n{text}")

            summary["pages"].append(
                {
                    "page_number": page_number,
                    "width": page.width,
                    "height": page.height,
                    "text_characters": len(text),
                    "word_count": len(words),
                    "char_count": len(page.chars),
                    "line_count": len(page.lines),
                    "rect_count": len(page.rects),
                    "curve_count": len(page.curves),
                    "image_count": len(page.images),
                    "table_count": len(tables),
                }
            )

            for table_index, table in enumerate(tables, start=1):
                table_name = f"{prefix}_table_{table_index:02d}"
                write_json(tables_dir / f"{table_name}.json", table)
                write_table_csv(tables_dir / f"{table_name}.csv", table)

    (extraction_dir / "document_text.txt").write_text(
        "".join(document_text).strip() + "\n",
        encoding="utf-8",
    )
    write_json(extraction_dir / "summary.json", summary)

    return extraction_dir
