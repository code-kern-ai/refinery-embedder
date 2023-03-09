# -*- coding: utf-8 -*-
from fastapi import FastAPI, responses, status
import controller
from data import data_type
from typing import List, Dict, Tuple
import torch

from submodules.model.business_objects import general
from util import request_util, config_handler

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
        },
        {
            "config_string": "bert-base-uncased",
            "description": "Generic embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        {
            "config_string": "roberta-base",
            "description": "Generic embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        {
            "config_string": "symanto/xlm-roberta-base-snli-mnli-anli-xnli",
            "description": "Few-shot optimimized embedding for English texts",
            "tokenizers": ["en_core_web_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        ### German ###
        {
            "config_string": "bert-base-german-cased",
            "description": "Generic transformer for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        {
            "config_string": "deepset/gbert-base",
            "description": "Generic transformer for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        {
            "config_string": "oliverguhr/german-sentiment-bert",
            "description": "Sentiment classification for German texts",
            "tokenizers": ["de_core_news_sm"],
            "applicability": {"attribute": True, "token": True},
        },
        ### Generic ###
        {
            "config_string": "bag-of-characters",
            "description": "Simple count of characters",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": True},
        },
        {
            "config_string": "bag-of-words",
            "description": "Simple count of words; PCA reduced",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
        },
        {
            "config_string": "tf-idf",
            "description": "Term frequency - inverse document frequency; PCA reduced",
            "tokenizers": ["all"],
            "applicability": {"attribute": True, "token": False},
        },
    ]

    return responses.JSONResponse(status_code=status.HTTP_200_OK, content=recommends)


@app.post("/classification/encode")
def encode_classification(request: data_type.Request) -> responses.PlainTextResponse:
    # session logic for threads in side
    status_code = controller.start_encoding_thread(request, "classification")

    return responses.PlainTextResponse(status_code=status_code)


@app.post("/extraction/encode")
def encode_extraction(request: data_type.Request) -> responses.PlainTextResponse:
    # session logic for threads in side
    status_code = controller.start_encoding_thread(request, "extraction")
    return responses.PlainTextResponse(status_code=status_code)


@app.delete("/delete/{project_id}/{embedding_id}")
def delete_embedding(project_id: str, embedding_id: str) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    status_code = controller.delete_embedding(project_id, embedding_id)
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status_code)


@app.post("/upload_tensor_data/{project_id}/{embedding_id}")
def upload_tensor_data(
    project_id: str, embedding_id: str
) -> responses.PlainTextResponse:
    session_token = general.get_ctx_token()
    controller.upload_embedding_as_file(project_id, embedding_id)
    request_util.post_embedding_to_neural_search(project_id, embedding_id)
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.put("/config_changed")
def config_changed() -> responses.PlainTextResponse:
    config_handler.refresh_config()
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    text = ""
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        error_name = database_test.get("error")
        text += f"database_error:{error_name}:"

    if not text:
        text = "OK"
    return responses.PlainTextResponse(text)
