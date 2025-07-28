from typing import List, Optional, Union, Generator
from sentence_transformers import SentenceTransformer
from src.embedders import util
from src.embedders.classification import SentenceEmbedder
from src.util import request_util
from spacy.tokens.doc import Doc
import torch
from openai import OpenAI, AzureOpenAI
from openai import AuthenticationError, RateLimitError
import time


class TransformerSentenceEmbedder(SentenceEmbedder):
    """Embeds documents using large, pre-trained transformers from https://huggingface.co

    Args:
        config_string (str): Name of the model listed on https://huggingface.co/models
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(self, config_string: str, batch_size: int = 128):
        super().__init__(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(config_string).to(self.device)

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            yield self.model.encode(documents_batch, show_progress_bar=False).tolist()


class HuggingFaceSentenceEmbedder(TransformerSentenceEmbedder):
    def __init__(self, config_string: str, batch_size: int = 128):
        super().__init__(config_string, batch_size)

    @staticmethod
    def load(embedder: dict) -> "HuggingFaceSentenceEmbedder":
        return HuggingFaceSentenceEmbedder(
            config_string=request_util.get_model_path(embedder["config_string"]),
            batch_size=embedder["batch_size"],
        )

    def to_json(self) -> dict:
        return {
            "cls": "HuggingFaceSentenceEmbedder",
            "model_name": self.model.model_card_data.base_model,
            "batch_size": self.batch_size,
        }

    def dump(self, project_id: str, embedding_id: str) -> None:
        export_file = util.INFERENCE_DIR / project_id / embedding_id / "embedder.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        util.write_json(self.to_json(), export_file, indent=2)


class OpenAISentenceEmbedder(SentenceEmbedder):
    def __init__(
        self,
        openai_api_key: str,
        model_name: str,
        batch_size: int = 128,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Embeds documents using large language models from https://openai.com or https://azure.microsoft.com

        Args:
            openai_api_key (str): API key from OpenAI or Azure
            model_name (str): Name of the embedding model from OpenAI (e.g. text-embedding-ada-002) or the name of your Azure endpoint
            batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
            api_base (str, optional): If you use Azure, you need to provide the base URL of your Azure endpoint (e.g. 'https://azureopenkernai.openai.azure.com/'). Defaults to None.
            api_type (str, optional): If you use Azure, you need to provide the type of your Azure endpoint (e.g. 'azure'). Defaults to None.
            api_version (str, optional): If you use Azure, you need to provide the version of your Azure endpoint (e.g. '2023-05-15'). Defaults to None.

        Raises:
            Exception: If you use Azure, you need to provide api_type, api_version and api_base.

        Examples:
            >>> from embedders.classification.contextual import OpenAISentenceEmbedder
            >>> embedder_openai = OpenAISentenceEmbedder(
            ...     "my-key-from-openai",
            ...     "text-embedding-ada-002",
            ... )
            >>> embeddings = embedder_openai.transform(["This is a test", "This is another test"])
            >>> print(embeddings)
            [[-0.0001, 0.0002, ...], [-0.0001, 0.0002, ...]]

            >>> from embedders.classification.contextual import OpenAISentenceEmbedder
            >>> embedder_azure = OpenAISentenceEmbedder(
            ...     "my-key-from-azure",
            ...     "my-endpoint-name",
            ...     api_base="https://azureopenkernai.openai.azure.com/",
            ...     api_type="azure",
            ...     api_version="2023-05-15",
            ... )
            >>> embeddings = embedder_azure.transform(["This is a test", "This is another test"])
            >>> print(embeddings)
            [[-0.0001, 0.0002, ...], [-0.0001, 0.0002, ...]]

        """
        super().__init__(batch_size)
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.api_base = api_base
        self.api_type = api_type
        self.api_version = api_version

        self.use_azure = any(
            [
                api_base is not None,
                api_type is not None,
                api_version is not None,
            ]
        )
        if self.use_azure:
            assert (
                api_type is not None
                and api_version is not None
                and api_base is not None
            ), "If you want to use Azure, you need to provide api_type, api_version and api_base."

    @property
    def openai_client(self):
        if self.use_azure:
            return AzureOpenAI(
                api_key=self.openai_api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version,
            )
        return OpenAI(
            api_key=self.openai_api_key,
            base_url=self.api_base,
        )

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch = [doc.replace("\n", " ") for doc in documents_batch]
            try:
                if self.use_azure:
                    embeddings = []
                    for azure_batch in util.batch(documents_batch, 16):
                        # azure only allows up to 16 documents per request
                        count = 0
                        while True and count < 60:
                            try:
                                count += 1
                                response = self.openai_client.embeddings.create(
                                    input=azure_batch, model=self.model_name
                                )
                                break
                            except RateLimitError as e:
                                if count >= 60:
                                    raise e
                                if count == 1:
                                    print(
                                        "Rate limit exceeded. Waiting 10 seconds...",
                                        flush=True,
                                    )
                                    time.sleep(10.05)
                                else:
                                    time.sleep(1)
                        embeddings += [entry.embedding for entry in response.data]
                else:
                    response = self.openai_client.embeddings.create(
                        input=documents_batch, model=self.model_name
                    )
                    embeddings = [entry.embedding for entry in response.data]
                yield embeddings
            except AuthenticationError:
                raise Exception(
                    "OpenAI API key is invalid. Please provide a valid API key in the constructor of OpenAISentenceEmbedder."
                )

    @staticmethod
    def load(embedder: dict) -> "OpenAISentenceEmbedder":
        return OpenAISentenceEmbedder(
            model_name=embedder["model_name"],
            batch_size=embedder["batch_size"],
            openai_api_key=embedder["openai_api_key"],
            api_base=embedder["api_base"],
            api_type=embedder["api_type"],
            api_version=embedder["api_version"],
        )

    def to_json(self) -> dict:
        return {
            "cls": "OpenAISentenceEmbedder",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "openai_api_key": self.openai_api_key,
            "api_base": self.api_base,
            "api_type": self.api_type,
            "api_version": self.api_version,
            "use_azure": self.use_azure,
        }

    def dump(self, project_id: str, embedding_id: str) -> None:
        export_file = util.INFERENCE_DIR / project_id / embedding_id / "embedder.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        util.write_json(self.to_json(), export_file, indent=2)
