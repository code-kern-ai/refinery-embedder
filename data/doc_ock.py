import os
import requests
import re

from dataclasses import dataclass
from submodules.model import enums
from typing import Any

from util import daemon

BASE_URI = os.getenv("DOC_OCK")


@dataclass
class Event:
    ConfigString: str
    State: str

    @classmethod
    def event_name(cls) -> str:
        # transforms the class name so that it is better readable in MixPanel,
        # e.g. CreateProject becomes "Create Project"
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", cls.__name__
        )
        return " ".join([m.group(0) for m in matches])


def post_embedding_creation(user_id: str, config_string: str) -> Any:
    return __post_event_threaded(user_id, config_string, enums.EmbeddingState.INITIALIZING.value)


def post_embedding_encoding(user_id: str, config_string: str) -> Any:
    return __post_event_threaded(user_id, config_string, enums.EmbeddingState.ENCODING.value)


def post_embedding_finished(user_id: str, config_string: str) -> Any:
    return __post_event_threaded(user_id, config_string, enums.EmbeddingState.FINISHED.value)


def post_embedding_failed(user_id: str, config_string: str) -> Any:
    return __post_event_threaded(user_id, config_string, enums.EmbeddingState.FAILED.value)


def __post_event_threaded(user_id: str, config_string: str, state: str) -> Any:
    daemon.run(__post_event, user_id, config_string, state)

def __post_event(user_id: str, config_string: str, state: str) -> Any:
    try:
        if not user_id:
            return  # migration is without user id (None)
        url = f"{BASE_URI}/track/{user_id}/Create Embedding"
        data = {
            "ConfigString": config_string,
            "State": state,
            "Host": os.getenv("S3_ENDPOINT"),
        }

        response = requests.post(url, json=data)

        if response.status_code != 200:
            raise Exception("Could not send data to Doc Ock")

        if response.headers.get("content-type") == "application/json":
            return response.json()
        else:
            return response.text
    except Exception as e:
        print("Sending of message failed.", str(e), flush=True)