# -*- coding: utf-8 -*-
"""
Resolve embedder class names to classes without using eval().
Used when loading serialized embedder configs (e.g. from JSON).
"""

from typing import Any, Type

_ALLOWED_EMBEDDER_CLASSES = frozenset({
    "HuggingFaceSentenceEmbedder",
    "OpenAISentenceEmbedder",
    "PrivatemodeAISentenceEmbedder",
    "PCASentenceReducer",
})


def get_embedder_class(name: str) -> Type[Any]:
    """
    Return the embedder class for the given name.
    Only allows known embedder class names to prevent code injection.
    """
    if name not in _ALLOWED_EMBEDDER_CLASSES:
        raise ValueError(f"Unknown embedder class: {name!r}")
    if name == "PCASentenceReducer":
        from src.embedders.classification.reduce import PCASentenceReducer
        return PCASentenceReducer
    if name == "HuggingFaceSentenceEmbedder":
        from src.embedders.classification.contextual import HuggingFaceSentenceEmbedder
        return HuggingFaceSentenceEmbedder
    if name == "OpenAISentenceEmbedder":
        from src.embedders.classification.contextual import OpenAISentenceEmbedder
        return OpenAISentenceEmbedder
    if name == "PrivatemodeAISentenceEmbedder":
        from src.embedders.classification.contextual import PrivatemodeAISentenceEmbedder
        return PrivatemodeAISentenceEmbedder
    raise ValueError(f"Unknown embedder class: {name!r}")
