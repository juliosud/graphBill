from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - runtime dependency check
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None
    load_dotenv = None


DEFAULT_EXTRACTION_DIR = Path("documents") / "txt" / "TELECOM WHOLESALE SERVICES INVOICE"
DEFAULT_OUTPUT_FILE = "reconstructed_invoice.txt"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_LAYOUT_WIDTH = 120


@dataclass(frozen=True)
class CoordinateItem:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float | None = None
    fontname: str | None = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_langchain() -> None:
    if (
        ChatOpenAI is None
        or ChatPromptTemplate is None
        or StrOutputParser is None
        or load_dotenv is None
    ):
        raise RuntimeError(
            "Missing LangChain dependencies. Install them with: "
            "py -m pip install -r requirements.txt"
        )


def get_words(page: dict[str, Any]) -> list[CoordinateItem]:
    items: list[CoordinateItem] = []
    for word in page.get("words", []):
        text = str(word.get("text", "")).strip()
        if not text:
            continue

        items.append(
            CoordinateItem(
                text=text,
                x0=float(word.get("x0", 0)),
                x1=float(word.get("x1", 0)),
                top=float(word.get("top", 0)),
                bottom=float(word.get("bottom", 0)),
                size=float(word["size"]) if word.get("size") is not None else None,
                fontname=word.get("fontname"),
            )
        )

    return sorted(items, key=lambda item: (round(item.top, 1), item.x0))


def group_rows(items: list[CoordinateItem], y_tolerance: float) -> list[list[CoordinateItem]]:
    rows: list[list[CoordinateItem]] = []

    for item in items:
        if not rows:
            rows.append([item])
            continue

        row_top = sum(existing.top for existing in rows[-1]) / len(rows[-1])
        if abs(item.top - row_top) <= y_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    return [sorted(row, key=lambda item: item.x0) for row in rows]


def row_y(row: list[CoordinateItem]) -> float:
    return sum(item.top for item in row) / len(row)


def page_width(page: dict[str, Any]) -> float:
    width = page.get("width") or 612
    return float(width)


def place_text(line: list[str], start: int, text: str) -> None:
    if start >= len(line):
        return

    for offset, char in enumerate(text):
        index = start + offset
        if index >= len(line):
            break
        line[index] = char


def build_visual_layout(
    page: dict[str, Any],
    rows: list[list[CoordinateItem]],
    layout_width: int,
) -> str:
    """Render a fixed-width approximation of the page using PDF x/y coordinates."""
    width = max(page_width(page), 1)
    output: list[str] = []
    previous_y: float | None = None

    for row in rows:
        if not row:
            continue

        current_y = row_y(row)
        if previous_y is not None:
            gap = current_y - previous_y
            if gap > 24:
                output.append("")

        line = [" "] * layout_width
        for item in row:
            column = round((item.x0 / width) * (layout_width - 1))
            place_text(line, max(column, 0), item.text.strip())

        output.append("".join(line).rstrip())
        previous_y = current_y

    return "\n".join(output)


def cluster_column_starts(
    items: list[CoordinateItem],
    tolerance: float,
    min_items: int,
) -> list[float]:
    clusters: list[list[float]] = []

    for item in sorted(items, key=lambda candidate: candidate.x0):
        for cluster in clusters:
            center = sum(cluster) / len(cluster)
            if abs(item.x0 - center) <= tolerance:
                cluster.append(item.x0)
                break
        else:
            clusters.append([item.x0])

    starts = [sum(cluster) / len(cluster) for cluster in clusters if len(cluster) >= min_items]
    return sorted(starts)


def column_boundaries(starts: list[float], width: float) -> list[tuple[float, float]]:
    if not starts:
        return []

    boundaries: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        left = 0 if index == 0 else (starts[index - 1] + start) / 2
        right = width if index == len(starts) - 1 else (start + starts[index + 1]) / 2
        boundaries.append((left, right))

    return boundaries


def assign_column(item: CoordinateItem, boundaries: list[tuple[float, float]]) -> int | None:
    midpoint = (item.x0 + item.x1) / 2
    for index, (left, right) in enumerate(boundaries):
        if left <= midpoint < right:
            return index
    return len(boundaries) - 1 if boundaries else None


def build_column_context(
    page: dict[str, Any],
    rows: list[list[CoordinateItem]],
    min_column_items: int = 3,
) -> str:
    words = [item for row in rows for item in row]
    starts = cluster_column_starts(words, tolerance=4.0, min_items=min_column_items)
    boundaries = column_boundaries(starts, page_width(page))
    if not starts or not boundaries:
        return "No stable column structure detected."

    lines = ["Detected column guides from repeated x positions:"]
    for index, (start, bounds) in enumerate(zip(starts, boundaries), start=1):
        left, right = bounds
        lines.append(f"- C{index}: x~{start:.1f}, range {left:.1f}-{right:.1f}")

    lines.append("")
    lines.append("Rows assigned to detected columns:")
    for row_number, row in enumerate(rows, start=1):
        cells = [""] * len(boundaries)
        for item in row:
            column_index = assign_column(item, boundaries)
            if column_index is None:
                continue
            cells[column_index] = " ".join(part for part in [cells[column_index], item.text.strip()] if part)

        if any(cells):
            rendered_cells = " | ".join(cells)
            lines.append(f"row {row_number:03d} y={row_y(row):.1f}: | {rendered_cells} |")

    return "\n".join(lines)


def build_coordinate_text(page: dict[str, Any], y_tolerance: float) -> str:
    rows = group_rows(get_words(page), y_tolerance)
    lines: list[str] = []

    for row_number, row in enumerate(rows, start=1):
        if not row:
            continue

        y = sum(item.top for item in row) / len(row)
        parts = [
            f"[x0={item.x0:.1f}, x1={item.x1:.1f}, y={item.top:.1f}] {item.text}"
            for item in row
        ]
        lines.append(f"row {row_number:03d} y={y:.1f}: " + " | ".join(parts))

    return "\n".join(lines)


def build_layout_context(
    page: dict[str, Any],
    y_tolerance: float,
    layout_width: int,
) -> dict[str, str]:
    rows = group_rows(get_words(page), y_tolerance)
    return {
        "visual_layout": build_visual_layout(page, rows, layout_width),
        "column_context": build_column_context(page, rows),
        "coordinate_rows": build_coordinate_text(page, y_tolerance),
    }


def page_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert invoice reconstruction agent. Rebuild the invoice "
                "page as clean plain text for use as context by another AI agent. Your "
                "top priority is layout fidelity: preserve the original reading order, "
                "section placement, page boundaries, column relationships, and table "
                "structure. Use the fixed-width visual layout as the source of truth for "
                "where text appears. Use the column guides to rebuild tables with stable "
                "columns and merged multi-line cells. Preserve factual text exactly. Do "
                "not invent missing values, totals, labels, or inferred fields. Prefer "
                "Markdown tables for tabular invoice lines, but keep nearby labels, page "
                "headers, and totals in their original relative locations.",
            ),
            (
                "human",
                "Reconstruct page {page_number} of {page_count}.\n\n"
                "Page size: width={width}, height={height}\n\n"
                "Raw pdfplumber text:\n{raw_text}\n\n"
                "Fixed-width visual layout reconstructed from coordinates:\n"
                "```text\n{visual_layout}\n```\n\n"
                "Column and row structure hints:\n{column_context}\n\n"
                "Coordinate rows:\n{coordinate_rows}\n\n"
                "Return only the reconstructed page text. Keep the output faithful to "
                "the invoice page layout, and structure all invoice line items as a "
                "complete table when the columns are visible.",
            ),
        ]
    )


def assembly_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an invoice document assembly agent. Combine reconstructed "
                "pages into one clean text document for downstream agent context. "
                "Preserve page boundaries, repeated page markers, invoice table order, "
                "and the original relationship between labels and values. Remove only "
                "obvious duplicate document titles when they add no value. Keep all "
                "invoice facts, line items, amounts, dates, IDs, and totals. Do not "
                "invent information.",
            ),
            (
                "human",
                "Assemble this reconstructed invoice into one final context document.\n\n"
                "{pages_text}\n\n"
                "Return only the final reconstructed invoice text.",
            ),
        ]
    )


def make_llm(model: str, temperature: float) -> ChatOpenAI:
    require_langchain()
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(model=model, temperature=temperature)


def reconstruct_pages(
    extraction_dir: Path,
    model: str,
    temperature: float,
    y_tolerance: float,
    layout_width: int,
) -> list[str]:
    pages_dir = extraction_dir / "pages"
    summary_path = extraction_dir / "summary.json"

    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages folder not found: {pages_dir}")

    page_paths = sorted(pages_dir.glob("page_*.json"))
    if not page_paths:
        raise FileNotFoundError(f"No page JSON files found in: {pages_dir}")

    summary = load_json(summary_path) if summary_path.exists() else {}
    page_count = int(summary.get("page_count") or len(page_paths))

    llm = make_llm(model=model, temperature=temperature)
    chain = page_prompt() | llm | StrOutputParser()

    reconstructed_pages: list[str] = []
    for page_path in page_paths:
        page = load_json(page_path)
        page_number = page.get("page_number", len(reconstructed_pages) + 1)
        print(f"Reconstructing page {page_number}/{page_count}...")
        layout_context = build_layout_context(page, y_tolerance, layout_width)

        reconstructed = chain.invoke(
            {
                "page_number": page_number,
                "page_count": page_count,
                "width": page.get("width"),
                "height": page.get("height"),
                "raw_text": page.get("text", ""),
                "visual_layout": layout_context["visual_layout"],
                "column_context": layout_context["column_context"],
                "coordinate_rows": layout_context["coordinate_rows"],
            }
        ).strip()

        reconstructed_pages.append(f"--- Reconstructed Page {page_number} ---\n{reconstructed}")

    return reconstructed_pages


def assemble_document(
    pages: list[str],
    model: str,
    temperature: float,
    skip_final_agent: bool,
) -> str:
    pages_text = "\n\n".join(pages)
    if skip_final_agent:
        return pages_text.strip() + "\n"

    llm = make_llm(model=model, temperature=temperature)
    chain = assembly_prompt() | llm | StrOutputParser()
    print("Assembling final reconstructed invoice...")
    return chain.invoke({"pages_text": pages_text}).strip() + "\n"


def reconstruct_invoice(
    extraction_dir: Path,
    output: str | Path = DEFAULT_OUTPUT_FILE,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    y_tolerance: float = 3.0,
    layout_width: int = DEFAULT_LAYOUT_WIDTH,
    skip_final_agent: bool = False,
) -> Path:
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = extraction_dir / output_path

    pages = reconstruct_pages(
        extraction_dir=extraction_dir,
        model=model,
        temperature=temperature,
        y_tolerance=y_tolerance,
        layout_width=layout_width,
    )
    final_text = assemble_document(
        pages=pages,
        model=model,
        temperature=temperature,
        skip_final_agent=skip_final_agent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_text, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use LangChain and a GPT model to reconstruct an invoice from "
            "pdfplumber x/y coordinate JSON."
        )
    )
    parser.add_argument(
        "--extraction-dir",
        default=str(DEFAULT_EXTRACTION_DIR),
        help=f"Folder created by extract_pdfplumber.py. Defaults to: {DEFAULT_EXTRACTION_DIR}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output text file name/path. Defaults to: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name to use through LangChain. Defaults to: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Model temperature. Defaults to 0 for deterministic reconstruction.",
    )
    parser.add_argument(
        "--y-tolerance",
        type=float,
        default=3.0,
        help="Maximum y-coordinate difference for grouping words into a row.",
    )
    parser.add_argument(
        "--layout-width",
        type=int,
        default=DEFAULT_LAYOUT_WIDTH,
        help=(
            "Character width for the fixed-width visual layout sent to the model. "
            f"Defaults to {DEFAULT_LAYOUT_WIDTH}."
        ),
    )
    parser.add_argument(
        "--skip-final-agent",
        action="store_true",
        help="Write page reconstructions directly without the final assembly pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extraction_dir = Path(args.extraction_dir)

    try:
        output_path = reconstruct_invoice(
            extraction_dir=extraction_dir,
            output=args.output,
            model=args.model,
            temperature=args.temperature,
            y_tolerance=args.y_tolerance,
            layout_width=args.layout_width,
            skip_final_agent=args.skip_final_agent,
        )
    except Exception as exc:
        print(f"Invoice reconstruction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Reconstructed invoice written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
