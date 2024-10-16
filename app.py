# -*- coding: utf-8 -*-
from fastapi import FastAPI, responses, status, Request
import controller
from data import data_type
from typing import Union
import torch

from submodules.model.business_objects import general
from util import request_util, config_handler
from submodules.model import session

app = FastAPI()

if torch.cuda.is_available():
    print(
        f"--- Running with GPU acceleration: {torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )
else:
    print(
        "--- Running on CPU. If you're facing performance issues, you should consider switching to a CUDA device",
        flush=True,
    )


@app.middleware("http")
async def handle_db_session(request: Request, call_next):
    session_token = general.get_ctx_token()

    request.state.session_token = session_token
    try:
        response = await call_next(request)
    finally:
        general.remove_and_refresh_session(session_token)

    return response


@app.get("/classification/recommend/{data_type}")
def recommendations(
    data_type: str,
) -> responses.JSONResponse:
    recommends = [
        ### English ###
        {
            "config_string": "distilbert-base-uncased",
            "description": "Lightweight generic embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        {
            "config_string": "bert-base-uncased",
            "description": "Generic embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        {
            "config_string": "roberta-base",
            "description": "Generic embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        {
            "config_string": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
            "description": "Few-shot optimimized embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        ### German ###
        {
            "config_string": "bert-base-german-cased",
            "description": "Generic transformer for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        {
            "config_string": "deepset/gbert-base",
            "description": "Generic transformer for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        {
            "config_string": "oliverguhr/german-sentiment-bert",
            "description": "Sentiment classification for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
            "platform": "huggingface",
        },
        ### Generic ###
        {
            "config_string": "bag-of-characters",
            "description": "Simple count of characters",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": True},
            "platform": "python",
        },
        {
            "config_string": "bag-of-words",
            "description": "Simple count of words; PCA reduced",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
            "platform": "python",
        },
        {
            "config_string": "tf-idf",
            "description": "Term frequency - inverse document frequency; PCA reduced",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
            "platform": "python",
        },
        {
            "config_string": "text-embedding-3-small",
            "description": "Cheap and reliable transformer",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
            "platform": "openai",
        },
        {
            "config_string": "text-embedding-3-large",
            "description": "Slower and more expensive model with better overall performance",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
            "platform": "openai",
        },
        {
            "config_string": "text-embedding-ada-002",
            "description": "Most common used openai transformer (outdated)",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
            "platform": "openai",
        },
    ]

    return responses.JSONResponse(status_code=status.HTTP_200_OK, content=recommends)


@app.post("/embed")
def embed(request: data_type.EmbeddingRequest) -> responses.PlainTextResponse:
    status_code = controller.manage_encoding_thread(
        request.project_id, request.embedding_id
    )
    return responses.PlainTextResponse(status_code=status_code)


@app.delete("/delete/{project_id}/{embedding_id}")
def delete_embedding(project_id: str, embedding_id: str) -> responses.PlainTextResponse:
    status_code = controller.delete_embedding(project_id, embedding_id)
    return responses.PlainTextResponse(status_code=status_code)


@app.post("/upload_tensor_data/{project_id}/{embedding_id}")
def upload_tensor_data(
    project_id: str, embedding_id: str
) -> responses.PlainTextResponse:
    controller.upload_embedding_as_file(project_id, embedding_id)
    request_util.post_embedding_to_neural_search(project_id, embedding_id)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/re_embed_records/{project_id}")
def re_embed_record(
    project_id: str, request: data_type.EmbeddingRebuildRequest
) -> responses.PlainTextResponse:
    controller.re_embed_records(project_id, request.changes)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.post("/calc-tensor-by-pkl/{project_id}/{embedding_id}")
def calc_tensor(
    project_id: str, embedding_id: str, request: data_type.EmbeddingCalcTensorByPkl
) -> Union[responses.PlainTextResponse, responses.PlainTextResponse]:
    if tensor := controller.calc_tensors(project_id, embedding_id, request.texts):
        return responses.JSONResponse(
            status_code=status.HTTP_200_OK, content={"tensor": tensor}
        )
    return responses.PlainTextResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content="Error while calculating tensor",
    )


@app.put("/config_changed")
def config_changed() -> responses.PlainTextResponse:
    config_handler.refresh_config()
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    text = ""
    status_code = status.HTTP_200_OK
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        error_name = database_test.get("error")
        text += f"database_error:{error_name}:"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if not text:
        text = "OK"
    return responses.PlainTextResponse(text, status_code=status_code)

session.start_session_cleanup_thread()