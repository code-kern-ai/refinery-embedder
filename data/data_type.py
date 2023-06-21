from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    project_id: str
    embedding_id: str

