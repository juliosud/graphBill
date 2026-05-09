__all__ = ["GraphRAGAgent"]


def __getattr__(name: str):
    if name == "GraphRAGAgent":
        from agents.graphrag_agent import GraphRAGAgent

        return GraphRAGAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

