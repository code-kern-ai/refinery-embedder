import requests
import os

from embedders.enums import WarningType

from submodules.model.business_objects import project

embedding_warning_templates = {
    WarningType.DOCUMENT_IS_SPLITTED.value: (
        "For {record_number} records, the text length exceeds the model's max input"
        " length. For these records, the texts are splitted and the parts are processed"
        " individually. For example, record {example_record_msg}."
    ),
    WarningType.TOKEN_MISMATCHING.value: (
        "For {record_number} records, the number of embeddings does not match the "
        "number of spacy tokens. Please contact support.For example, record "
        "{example_record_msg}."
    ),
}


def send_project_update(project_id: str, message: str, is_global: bool = False) -> None:
    endpoint = os.getenv("WS_NOTIFY_ENDPOINT")
    if not endpoint:
        print(
            "- WS_NOTIFY_ENDPOINT not set -- did you run the start script?", flush=True
        )
        return

    if is_global:
        message = f"GLOBAL:{message}"
    else:
        message = f"{project_id}:{message}"
    project_item = project.get(project_id)
    organization_id = str(project_item.organization_id)
    req = requests.post(
        f"{endpoint}/notify",
        json={
            "organization": organization_id,
            "message": message,
        },
    )
    if req.status_code != 200:
        print("Could not send notification update", flush=True)
