from typing import Any, Generator, List
from pathlib import Path
import numpy as np
import os
import json
import pickle

INFERENCE_DIR = Path(os.getenv("INFERENCE_DIR", "/inference"))


def batch(documents: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    length = len(documents)
    for idx in range(0, length, batch_size):
        yield documents[idx : min(idx + batch_size, length)]


def num_batches(documents: List[Any], batch_size: int) -> int:
    return int(np.ceil(len(documents) / batch_size))


def read_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pickle(obj: Any, file_path: str, **kwargs) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, **kwargs)


def read_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(obj: Any, file_path: str, **kwargs) -> None:
    with open(file_path, "w") as f:
        json.dump(obj, f, **kwargs)
