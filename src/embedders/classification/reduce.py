from spacy.tokens.doc import Doc
from typing import Union, List, Generator
import numpy as np
import pickle
from src.embedders import PCAReducer, util

# Embedder imports are used by eval(Embedder) in load methods
from src.embedders.classification.contextual import (  # noqa: F401
    OpenAISentenceEmbedder,
    HuggingFaceSentenceEmbedder,
    PrivatemodeAISentenceEmbedder,
)


class PCASentenceReducer(PCAReducer):
    def _transform(
        self, embeddings: List[List[Union[int, float]]]
    ) -> List[List[Union[float, int]]]:
        return self.reducer.transform(embeddings).tolist()

    def _reduce(
        self,
        documents: List[Union[str, Doc]],
        as_generator: bool,
        fit_model: bool,
        fit_after_n_batches: int,
    ) -> Generator[List[List[Union[float, int]]], None, None]:
        if fit_model:
            embeddings_training = []
            num_batches = util.num_batches(documents, self.embedder.batch_size)
            fit_after_n_batches = min(num_batches, fit_after_n_batches) - 1
            for batch_idx, batch in enumerate(
                self.embedder.fit_transform(documents, as_generator)
            ):
                if batch_idx <= fit_after_n_batches:
                    embeddings_training.append(batch)

                if batch_idx == fit_after_n_batches:
                    embeddings_training_flattened = []
                    for batch_training in embeddings_training:
                        embeddings_training_flattened.extend(batch_training)
                    embeddings_training_flattened = np.array(
                        embeddings_training_flattened
                    )
                    if (
                        embeddings_training_flattened.shape[1]
                        < self.reducer.n_components
                        and self.autocorrect_n_components
                    ):
                        self.reducer.n_components = embeddings_training_flattened.shape[
                            1
                        ]
                    self.reducer.fit(embeddings_training_flattened)

                    for batch_training in embeddings_training:
                        yield self._transform(batch_training)
                if batch_idx > fit_after_n_batches:
                    yield self._transform(batch)
        else:
            if as_generator:
                embeddings = [
                    emb
                    for batch in self.embedder.transform(documents, as_generator)
                    for emb in batch
                ]
                yield from util.batch(self._transform(embeddings), self.batch_size)
            else:
                yield self._transform(embeddings)

    @staticmethod
    def load(embedder: dict) -> "PCASentenceReducer":
        reducer = pickle.loads(
            embedder["reducer_pkl_bytes"].encode("latin-1")
        )  # Decode to latin1 to avoid binary issues in JSON
        Embedder = eval(embedder["embedder"]["cls"])
        return PCASentenceReducer(
            embedder=Embedder.load(embedder["embedder"]),
            reducer=reducer,
        )

    def to_json(self) -> dict:
        return {
            "cls": "PCASentenceReducer",
            "embedder": self.embedder.to_json(),
            "reducer_pkl_bytes": pickle.dumps(self.reducer).decode(
                "latin-1"
            ),  # Encode to latin1 to avoid binary issues in JSON
        }

    def dump(self, project_id: str, embedding_id: str) -> None:
        export_file = util.INFERENCE_DIR / project_id / f"embedder-{embedding_id}.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        util.write_json(self.to_json(), export_file, indent=2)
