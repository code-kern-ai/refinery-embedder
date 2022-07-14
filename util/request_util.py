import os
import requests

NEURAL_SEARCH_BASE_URI = os.getenv("NEURAL_SEARCH")


def post_embedding_to_neural_search(project_id: str, embedding_id: str) -> None:
    url = f"{NEURAL_SEARCH_BASE_URI}/recreate_collection"
    params = {
        "project_id": project_id,
        "embedding_id": embedding_id,
    }
    requests.post(url, params=params)


def delete_embedding_from_neural_search(embedding_id: str) -> None:
    url = f"{NEURAL_SEARCH_BASE_URI}/delete_collection"
    params = {"embedding_id": embedding_id}
    requests.put(url, params=params)
