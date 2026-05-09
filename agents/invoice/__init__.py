__all__ = [
    "InvoiceExtractionOrchestrator",
    "InvoiceExtractionState",
    "extract_invoice_graph_with_forest",
    "extract_invoice_graph",
    "reconstruct_invoice",
]


def __getattr__(name: str):
    if name in {"InvoiceExtractionOrchestrator", "InvoiceExtractionState"}:
        from agents.invoice.invoice_extraction_orchestrator import (
            InvoiceExtractionOrchestrator,
            InvoiceExtractionState,
        )

        exports = {
            "InvoiceExtractionOrchestrator": InvoiceExtractionOrchestrator,
            "InvoiceExtractionState": InvoiceExtractionState,
        }
        return exports[name]

    if name == "extract_invoice_graph_with_forest":
        from agents.invoice.agentic_forest_graph_agent import extract_invoice_graph_with_forest

        return extract_invoice_graph_with_forest

    if name == "extract_invoice_graph":
        from agents.invoice.invoice_graph_extraction_agent import extract_invoice_graph

        return extract_invoice_graph

    if name == "reconstruct_invoice":
        from agents.invoice.reconstruct_invoice_agent import reconstruct_invoice

        return reconstruct_invoice

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
