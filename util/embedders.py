from embedders.classification.count_based import (
    BagOfCharsSentenceEmbedder,
    BagOfWordsSentenceEmbedder,
    TfidfSentenceEmbedder,
)
from embedders.classification.contextual import TransformerSentenceEmbedder
from embedders.extraction.count_based import BagOfCharsTokenEmbedder
from embedders.extraction.contextual import TransformerTokenEmbedder
from embedders.classification.reduce import PCASentenceReducer
from embedders.extraction.reduce import PCATokenReducer
from embedders import Transformer

from submodules.model.business_objects import record


def get_embedder(
    project_id: str,
    embedding_type: str,
    config_string: str,
    language_code: str,
) -> Transformer:
    if embedding_type == "classification":
        batch_size = 128
        n_components = 64
        if config_string == "bag-of-characters":
            return BagOfCharsSentenceEmbedder(batch_size=batch_size)
        elif config_string == "bag-of-words":
            embedder = BagOfWordsSentenceEmbedder(batch_size=batch_size)
        elif config_string == "tf-idf":
            embedder = TfidfSentenceEmbedder(batch_size=batch_size)
        else:
            embedder = TransformerSentenceEmbedder(
                config_string=config_string, batch_size=batch_size
            )

        if record.count(project_id) < n_components:
            return embedder
        else:
            return PCASentenceReducer(embedder, n_components=n_components)

    else:  # extraction
        batch_size = 32
        n_components = 16
        if config_string == "bag-of-characters":
            return BagOfCharsTokenEmbedder(
                language_code=language_code,
                precomputed_docs=True,
                batch_size=batch_size,
            )
        if config_string == "bag-of-words":
            return None
        if config_string == "tf-idf":
            return None
        else:
            return PCATokenReducer(
                TransformerTokenEmbedder(
                    config_string=config_string,
                    language_code=language_code,
                    precomputed_docs=True,
                    batch_size=batch_size,
                ),
                n_components=n_components,
            )
