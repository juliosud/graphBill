from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - runtime dependency check
    ChatOpenAI = None
    Tool = None
    create_agent = None
    load_dotenv = None


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_OUTPUT_FILE = "invoice_graph.json"
GRAPH_SCHEMA_VERSION = "invoice-graph-v2"
DEFAULT_USE_FOREST = True


TELECOM_INVOICE_DOMAIN_GUIDANCE = """
These documents are generally telecom wholesale invoices. Vendors may format them
differently, but they often describe communications services sold to one or more
customers/accounts over a billing period. Expect some combination of:
- Vendor/carrier, customer/account, invoice metadata, payment terms, remittance details.
- Voice services such as domestic/international termination, premium routes, toll-free,
  inbound/outbound traffic, DID/SIP trunking, or carrier connectivity.
- Messaging services such as A2P SMS, OTP/authentication messages, promotional traffic,
  short code/long code traffic, delivery fees, or platform access.
- Recurring charges, usage-based charges, one-time charges, adjustments, discounts,
  taxes, regulatory fees, surcharges, credits, subtotals, and total due.
- Quantities measured in minutes, messages, numbers, trunks, channels, seats, months,
  or generic units, with rates such as per minute, per message, monthly, or flat fee.

Future analysis goals:
- Query totals by vendor, customer, invoice, billing period, service category, country,
  route, service code, currency, unit, and charge type.
- Compare usage, rates, discounts, taxes/fees, and total spend across vendors and months.
- Preserve enough normalized fields for metrics while keeping source evidence.
- Keep original labels and descriptions too, because vendor terminology varies.

Preferred graph shape:
- Use the invoice as the document root: Invoice -> Vendor, Invoice -> InvoiceSummary,
  Invoice -> BillingPeriod, Invoice -> PaymentTerms, Invoice -> Currency, and
  Invoice -> RemittanceInstruction when those details exist.
- Put account and service detail under the vendor, not directly under the invoice:
  Vendor -> Account -> TelecomService -> UsageMeasurement/Rate/Charge/Discount/TaxOrFee.
- Connect accounts back to customers with Account -> Customer or Customer -> Account
  relationships, and preserve bill-to evidence.
- Treat each invoice line as a TelecomService plus child economics when possible. Keep a
  ServiceLineItem only as an optional source-line wrapper when the invoice row itself is
  useful to preserve.
- Put totals, subtotals, balances, taxes, credits, and due amounts in InvoiceSummary or
  PaymentSummary nodes and connect summaries to the invoice/account they summarize.

Extraction guidance:
- Use flexible graph entities; do not force fields that are absent.
- Normalize numeric amounts when possible, but preserve the original text and currency.
- For every charge-like line, try to capture: description, service category, quantity,
  unit, rate, discount, amount, currency, charge type, account/customer, and page evidence.
- Distinguish usage/rate/discount/charge/tax/fee/total when the invoice makes that clear.
- If a document contains multiple bill-to accounts, treat each account/customer section as
  its own subgraph under the vendor and link it to the relevant invoice summary.
- Prefer relationships that support analytics: vendor serves account, account has service,
  service has usage/rate/charge/discount/tax/fee, and summaries roll up invoice/account totals.
"""


def require_langchain_agent() -> None:
    if (
        ChatOpenAI is None
        or Tool is None
        or create_agent is None
        or load_dotenv is None
    ):
        raise RuntimeError(
            "Missing LangChain agent dependencies. Install them with: "
            "py -m pip install -r requirements.txt"
        )


def make_llm(model: str, temperature: float) -> ChatOpenAI:
    require_langchain_agent()
    load_dotenv(encoding="utf-8-sig")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(model=model, temperature=temperature)


def summarize_invoice_structure(invoice_text: str) -> str:
    lines = [line.rstrip() for line in invoice_text.splitlines()]
    non_empty = [line for line in lines if line.strip()]
    pages = [line for line in non_empty if line.strip().startswith("--- Page")]
    headings = [
        line.strip()
        for line in non_empty
        if line.strip().isupper() and len(line.strip()) > 3 and not line.strip().startswith("---")
    ]
    table_headers = [line.strip() for line in non_empty if line.strip().startswith("|") and "---" not in line]
    identifiers = sorted(
        set(re.findall(r"\b(?:Invoice Number|Customer ID|Vendor ID|VAT ID):\s*[^\n]+", invoice_text))
    )
    money_values = sorted(set(re.findall(r"[$€£]\s?\d[\d,]*(?:\.\d{2})?", invoice_text)))

    return json.dumps(
        {
            "page_count_markers": len(pages),
            "page_markers": pages[:20],
            "candidate_headings": headings[:50],
            "candidate_table_headers": table_headers[:20],
            "candidate_identifiers": identifiers[:50],
            "sample_money_values": money_values[:50],
            "line_count": len(lines),
        },
        ensure_ascii=False,
        indent=2,
    )


def graph_schema_guidance(_: str = "") -> str:
    return json.dumps(
        {
            "entity_types": [
                "Invoice",
                "InvoiceSummary",
                "Vendor",
                "Customer",
                "Account",
                "Address",
                "Identifier",
                "BillingPeriod",
                "PaymentTerms",
                "Currency",
                "ServiceCategory",
                "ServiceLineItem",
                "TelecomService",
                "UsageMeasurement",
                "Rate",
                "Discount",
                "Charge",
                "TaxOrFee",
                "Adjustment",
                "PaymentSummary",
                "RemittanceInstruction",
                "DocumentPage",
            ],
            "relationship_types": [
                "ISSUED_BY",
                "HAS_SUMMARY",
                "BILLED_TO",
                "BILLS_CUSTOMER",
                "SERVES_ACCOUNT",
                "HAS_ACCOUNT",
                "OWNED_BY",
                "HAS_IDENTIFIER",
                "HAS_ADDRESS",
                "HAS_BILLING_PERIOD",
                "HAS_PAYMENT_TERMS",
                "USES_CURRENCY",
                "CONTAINS_SERVICE_CATEGORY",
                "CONTAINS_LINE_ITEM",
                "USES_SERVICE",
                "HAS_USAGE",
                "HAS_RATE",
                "HAS_DISCOUNT",
                "HAS_CHARGE",
                "HAS_TAX_OR_FEE",
                "HAS_ADJUSTMENT",
                "HAS_REMITTANCE_INSTRUCTION",
                "SUMMARIZES",
                "ROLLS_UP_TO",
                "APPEARS_ON_PAGE",
                "NEXT_LINE_ITEM",
            ],
            "preferred_topology": [
                "Invoice -[:ISSUED_BY]-> Vendor",
                "Invoice -[:HAS_SUMMARY]-> InvoiceSummary",
                "Invoice -[:HAS_BILLING_PERIOD]-> BillingPeriod",
                "Invoice -[:HAS_PAYMENT_TERMS]-> PaymentTerms",
                "Invoice -[:USES_CURRENCY]-> Currency",
                "Vendor -[:BILLS_CUSTOMER]-> Customer",
                "Vendor -[:SERVES_ACCOUNT]-> Account",
                "Account -[:OWNED_BY]-> Customer",
                "Account -[:USES_SERVICE]-> TelecomService",
                "TelecomService -[:HAS_USAGE]-> UsageMeasurement",
                "TelecomService -[:HAS_RATE]-> Rate",
                "TelecomService -[:HAS_CHARGE]-> Charge",
                "TelecomService -[:HAS_DISCOUNT]-> Discount",
                "TelecomService -[:HAS_TAX_OR_FEE]-> TaxOrFee",
                "PaymentSummary -[:SUMMARIZES]-> Account",
                "PaymentSummary -[:ROLLS_UP_TO]-> InvoiceSummary",
            ],
            "metric_properties_to_prefer": [
                "amount",
                "currency",
                "quantity",
                "unit",
                "rate",
                "rate_unit",
                "discount_percent",
                "charge_type",
                "service_category",
                "service_code",
                "route",
                "country_or_region",
                "billing_period_start",
                "billing_period_end",
                "invoice_date",
                "due_date",
                "vendor_name",
                "customer_name",
                "account_id",
                "summary_type",
                "subtotal_amount",
                "tax_fee_total",
                "adjustment_total",
                "total_due",
            ],
            "json_contract": {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "source": {"txt_path": "string", "document_name": "string"},
                "analysis": {
                    "structure_summary": "string",
                    "extraction_plan": ["ordered steps used to extract the graph"],
                    "assumptions": ["only explicit, evidence-backed assumptions"],
                },
                "entities": [
                    {
                        "id": "stable unique id",
                        "type": "entity type",
                        "labels": ["graph labels"],
                        "properties": {},
                        "evidence": [{"page": "page marker or null", "text": "source snippet"}],
                    }
                ],
                "relationships": [
                    {
                        "id": "stable unique id",
                        "type": "relationship type",
                        "source_id": "entity id",
                        "target_id": "entity id",
                        "properties": {},
                        "evidence": [{"page": "page marker or null", "text": "source snippet"}],
                    }
                ],
                "hierarchy_rules": [
                    "One Invoice root per source document.",
                    "Attach document-level metadata and totals to Invoice through child summary/detail nodes.",
                    "Attach accounts to Vendor with SERVES_ACCOUNT and customers with BILLS_CUSTOMER/OWNED_BY.",
                    "Represent each billable service as TelecomService with child economic nodes.",
                    "Use Charge for monetary line charges, UsageMeasurement for quantities, Rate for prices, Discount for reductions, and TaxOrFee for taxes/surcharges/fees.",
                    "Use PaymentSummary or InvoiceSummary for account-level and invoice-level rollups.",
                ],
                "unresolved": ["items that need review"],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def build_tools(invoice_text: str) -> list[Any]:
    require_langchain_agent()
    return [
        Tool.from_function(
            name="inspect_invoice_structure",
            func=lambda _: summarize_invoice_structure(invoice_text),
            description=(
                "Inspect the reconstructed invoice text and return page markers, headings, "
                "table candidates, identifiers, and money values."
            ),
        ),
        Tool.from_function(
            name="get_graph_json_contract",
            func=graph_schema_guidance,
            description=(
                "Return the required graph JSON contract, allowed entity types, and "
                "relationship types for invoice graph extraction."
            ),
        ),
    ]


def graph_extraction_system_prompt() -> str:
    return f"""You are an invoice graph extraction agent using a ReAct-style tool loop.

Your mission:
1. Understand how the reconstructed invoice text is structured.
2. Make a careful extraction plan.
3. Extract all useful graph database entities and relationships into JSON.

General telecom invoice guidance:
{TELECOM_INVOICE_DOMAIN_GUIDANCE}

Use tools to inspect the structure and confirm the JSON contract before producing the final answer.

Rules:
- Return only valid JSON in your final response.
- Capture entities, relationships, and evidence snippets.
- Prefer explicit source text over inference.
- Use stable ids that are safe for graph database imports.
- Preserve page evidence when page markers are present.
- Build the graph around this hierarchy: Invoice -> Vendor; Invoice -> InvoiceSummary; Vendor -> Account; Account -> TelecomService; TelecomService -> UsageMeasurement/Rate/Charge/Discount/TaxOrFee.
- Include service lines, charges, usage, rates, discounts, taxes/fees, customers, accounts, vendor, invoice metadata, billing period, totals, and payment terms whenever present.
- Include analysis-friendly normalized properties where explicitly supported by the text, such as amount, currency, quantity, unit, rate, rate_unit, discount_percent, service_category, service_code, country_or_region, route, billing period dates, invoice date, due date, vendor, customer, and account identifiers.
- Prefer separate economic child nodes for usage, rate, charge, discount, tax/fee, adjustment, and summary values instead of storing everything only on one line-item node.
- Use InvoiceSummary and PaymentSummary nodes for totals, subtotals, prior balances, credits, taxes/fees totals, adjustments, and total due.
- Keep both normalized values and original source descriptions so invoices from different vendors remain comparable without losing vendor-specific terminology.
- Think ahead to graph database queries and metric analysis across vendors, customers, months, service categories, and charge types.
- Add uncertain or missing items to unresolved instead of inventing.
- Before the final JSON, use the available tools to inspect the structure and schema contract.
- Do not wrap the final JSON in Markdown.
"""


def graph_extraction_user_prompt(txt_path: Path, invoice_text: str) -> str:
    return f"""Create graph-ready JSON from this reconstructed invoice text.

Text path: {txt_path}
Document name: {txt_path.parent.name}

Required top-level JSON keys:
schema_version, source, analysis, entities, relationships, unresolved.

Reconstructed invoice text:
```text
{invoice_text}
```
"""


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def normalize_graph_payload(payload: dict[str, Any], txt_path: Path) -> dict[str, Any]:
    payload.setdefault("schema_version", GRAPH_SCHEMA_VERSION)
    payload.setdefault(
        "source",
        {
            "txt_path": str(txt_path),
            "document_name": txt_path.parent.name,
        },
    )
    payload.setdefault("analysis", {})
    payload.setdefault("entities", [])
    payload.setdefault("relationships", [])
    payload.setdefault("unresolved", [])
    return payload


def get_agent_output(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if messages:
        content = getattr(messages[-1], "content", "")
        if isinstance(content, list):
            return "\n".join(str(part) for part in content)
        return str(content)

    output = result.get("output")
    if output is not None:
        return str(output)

    return str(result)


def extract_invoice_graph_single(
    txt_path: Path,
    output: str | Path = DEFAULT_OUTPUT_FILE,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
) -> Path:
    txt_path = txt_path.resolve()
    invoice_text = txt_path.read_text(encoding="utf-8")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = txt_path.parent / output_path

    llm = make_llm(model=model, temperature=temperature)
    tools = build_tools(invoice_text)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=graph_extraction_system_prompt(),
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

    payload = extract_json_payload(get_agent_output(result))
    payload = normalize_graph_payload(payload, txt_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def extract_invoice_graph(
    txt_path: Path,
    output: str | Path = DEFAULT_OUTPUT_FILE,
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    use_forest: bool = DEFAULT_USE_FOREST,
) -> Path:
    if not use_forest:
        return extract_invoice_graph_single(
            txt_path=txt_path,
            output=output,
            model=model,
            temperature=temperature,
        )

    from agents.invoice.agentic_forest_graph_agent import extract_invoice_graph_with_forest

    artifacts = extract_invoice_graph_with_forest(
        txt_path=txt_path,
        output=output,
        model=model,
        temperature=temperature,
    )
    return artifacts.final_graph_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract graph-ready JSON from a reconstructed invoice text file."
    )
    parser.add_argument("txt", help="Path to reconstructed_invoice.txt")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file name/path. Defaults to: {DEFAULT_OUTPUT_FILE}",
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
        help="Model temperature. Defaults to 0 for deterministic extraction.",
    )
    parser.add_argument(
        "--disable-forest",
        action="store_true",
        help="Use the legacy single-agent extractor instead of the agentic forest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_path = extract_invoice_graph(
            txt_path=Path(args.txt),
            output=args.output,
            model=args.model,
            temperature=args.temperature,
            use_forest=not args.disable_forest,
        )
    except Exception as exc:
        print(f"Invoice graph extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Invoice graph JSON written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
