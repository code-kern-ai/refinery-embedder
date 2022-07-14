# -*- coding: utf-8 -*-
from fastapi import FastAPI
import controller
from data import data_type
from typing import List, Dict, Tuple
import torch

from submodules.model.business_objects import general
from util import request_util

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
) -> Tuple[List[Dict[str, str]], int]:
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
        # TODO: w2v currently doesn't work yet, @Johannes needs to fix this
        # {
        #     "config_string": "word2vec",
        #     "description": "SkipGram based word representations (token-granularity only)",
        # },
    ]

    return recommends, 200


@app.post("/classification/encode")
def encode_classification(request: data_type.Request) -> Tuple[int, str]:
    # session logic for threads in side
    return controller.start_encoding_thread(request, "classification"), ""


@app.post("/extraction/encode")
def encode_extraction(request: data_type.Request) -> Tuple[int, str]:
    # session logic for threads in side
    return controller.start_encoding_thread(request, "extraction"), ""


@app.delete("/delete/{project_id}/{embedding_id}")
def delete_embedding(project_id: str, embedding_id: str) -> Tuple[int, str]:
    session_token = general.get_ctx_token()
    return_value = controller.delete_embedding(project_id, embedding_id)
    general.remove_and_refresh_session(session_token)
    return return_value, ""


@app.post("/upload_tensor_data/{project_id}/{embedding_id}")
def upload_tensor_data(project_id: str, embedding_id: str) -> Tuple[int, str]:
    session_token = general.get_ctx_token()
    controller.upload_embedding_as_file(project_id, embedding_id)
    request_util.post_embedding_to_neural_search(project_id, embedding_id)
    general.remove_and_refresh_session(session_token)
    return 200, ""
