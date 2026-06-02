"""App-level singletons populated during Chainlit startup."""
from __future__ import annotations

compiled_graph = None


def get_graph():
    return compiled_graph
