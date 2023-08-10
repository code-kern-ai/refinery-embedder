from typing import Optional
from embedders.classification.count_based import (
    BagOfCharsSentenceEmbedder,
    BagOfWordsSentenceEmbedder,
    TfidfSentenceEmbedder,
)
from embedders.classification.contextual import (
    OpenAISentenceEmbedder,
    HuggingFaceSentenceEmbedder,
    CohereSentenceEmbedder,
)
from embedders.extraction.count_based import BagOfCharsTokenEmbedder
from embedders.extraction.contextual import TransformerTokenEmbedder
from embedders.classification.reduce import PCASentenceReducer
from embedders.extraction.reduce import PCATokenReducer
from embedders import Transformer
from submodules.model import enums

from submodules.model.business_objects import record


def get_embedder(
    project_id: str,
    embedding_type: str,
    language_code: str,
    platform: str,
    model: Optional[str] = None,
    api_token: Optional[str] = None,
    additional_data: Optional[str] = None,
) -> Transformer:
    if embedding_type == enums.EmbeddingType.ON_ATTRIBUTE.value:
        batch_size = 128
        n_components = 64
        if platform == enums.EmbeddingPlatform.PYTHON.value:
            if model == "bag-of-characters":
                return BagOfCharsSentenceEmbedder(batch_size=batch_size)
            elif model == "bag-of-words":
                embedder = BagOfWordsSentenceEmbedder(batch_size=batch_size)
            elif model == "tf-idf":
                embedder = TfidfSentenceEmbedder(batch_size=batch_size)
            else:
                raise Exception(f"Unknown model {model}")
        elif (
            platform == enums.EmbeddingPlatform.OPENAI.value
            or platform == enums.EmbeddingPlatform.AZURE.value
        ):
            # azure can only handle 16 per request however these are still batched to batch_size in embedders lib
            # base, type & version are needed only for Azure. Model field is used for model name (OpenAI) or engine (Azure)
            embedder = OpenAISentenceEmbedder(
                openai_api_key=api_token,
                model_name=model,
                batch_size=batch_size,
                api_base=additional_data.get("base"),
                api_type=additional_data.get("type"),
                api_version=additional_data.get("version"),
            )
        elif platform == enums.EmbeddingPlatform.HUGGINGFACE.value:
            embedder = HuggingFaceSentenceEmbedder(
                config_string=model, batch_size=batch_size
            )
        elif platform == enums.EmbeddingPlatform.COHERE.value:
            embedder = CohereSentenceEmbedder(
                cohere_api_key=api_token, batch_size=batch_size
            )
        else:
            raise Exception(f"Unknown platform {platform}")

        if record.count(project_id) < n_components:
            return embedder
        else:
            return PCASentenceReducer(embedder, n_components=n_components)

    else:  # extraction
        batch_size = 32
        n_components = 16
        if model == "bag-of-characters":
            return BagOfCharsTokenEmbedder(
                language_code=language_code,
                precomputed_docs=True,
                batch_size=batch_size,
            )
        if model == "bag-of-words":
            return None
        if model == "tf-idf":
            return None
        else:
            return PCATokenReducer(
                TransformerTokenEmbedder(
                    config_string=model,
                    language_code=language_code,
                    precomputed_docs=True,
                    batch_size=batch_size,
                ),
                n_components=n_components,
            )
