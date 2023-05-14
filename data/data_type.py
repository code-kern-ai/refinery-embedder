from pydantic import BaseModel


class Request(BaseModel):
    project_id: str
    attribute_id: str
    user_id: str
    config_string: str
    platform: str
