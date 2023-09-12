from typing import Dict, List
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    project_id: str
    embedding_id: str


class EmbeddingRebuildRequest(BaseModel):
    # example request structure:
    # {"<embedding_id>":[{"record_id":"<record_id>","attribute_name":"<attribute_name>","sub_key":<sub_key>}]}
    # note that sub_key is optional and only for embedding lists relevant
    # also sub_key is an int but converted to string in the request

    changes: Dict[str, List[Dict[str, str]]]
