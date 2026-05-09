from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invoice_ingestor.extraction import extract_pdf


DEFAULT_PDF = Path("documents") / "pdf" / "TELECOM WHOLESALE SERVICES INVOICE.pdf"
DEFAULT_OUTPUT_ROOT = Path("documents") / "txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a PDF with pdfplumber into a structured extraction folder."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(DEFAULT_PDF),
        help=f"PDF file to extract. Defaults to: {DEFAULT_PDF}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output folder. Defaults to: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        output_dir = extract_pdf(Path(args.pdf), Path(args.output))
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Extraction complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
