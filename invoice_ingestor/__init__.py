from invoice_ingestor.config import IngestionConfig

__all__ = ["IngestionConfig", "ingest_invoice"]


def __getattr__(name: str):
    if name == "ingest_invoice":
        from invoice_ingestor.pipeline import ingest_invoice

        return ingest_invoice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
