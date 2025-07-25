from spacy.tokens.doc import Doc
from typing import Union, List, Generator
import numpy as np
from src.embedders import PCAReducer, util
from src.embedders.classification.contextual import (
    OpenAISentenceEmbedder,
    HuggingFaceSentenceEmbedder,
)


class PCASentenceReducer(PCAReducer):
    def _transform(
        self, embeddings: List[List[Union[int, float]]]
    ) -> List[List[Union[float, int]]]:
        return self.reducer.transform(embeddings).tolist()

    def _reduce(
        self,
        documents: List[Union[str, Doc]],
        fit_model: bool,
        fit_after_n_batches: int,
    ) -> Generator[List[List[Union[float, int]]], None, None]:
        if fit_model:
            embeddings_training = []
            num_batches = util.num_batches(documents, self.embedder.batch_size)
            fit_after_n_batches = min(num_batches, fit_after_n_batches) - 1
            for batch_idx, batch in enumerate(
                self.embedder.fit_transform(documents, as_generator=True)
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
            embeddings = self.embedder.transform(documents)
            yield self._transform(embeddings)

    @staticmethod
    def load(embedder: dict) -> "PCASentenceReducer":
        reducer = util.read_pickle(embedder["reducer_pkl"])
        Embedder = eval(embedder["embedder"]["cls"])
        return PCASentenceReducer(
            embedder=Embedder.load(embedder["embedder"]),
            reducer=reducer,
        )

    def to_json(self) -> dict:
        return {
            "cls": "PCASentenceReducer",
            "embedder": self.embedder.to_json(),
        }

    def dump(self, project_id: str, embedding_id: str) -> None:
        export_file = util.INFERENCE_DIR / project_id / embedding_id / "embedder.json"
        export_file.parent.mkdir(parents=True, exist_ok=True)
        pkl_file = util.INFERENCE_DIR / project_id / embedding_id / "reducer.pkl"
        util.write_pickle(self.reducer, pkl_file)

        json_obj = self.to_json()
        json_obj["reducer_pkl"] = str(pkl_file)
        util.write_json(json_obj, export_file, indent=2)
