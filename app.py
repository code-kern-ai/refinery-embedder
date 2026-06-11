from contextlib import asynccontextmanager
from fastapi import FastAPI, responses, status, Request
from typing import AsyncGenerator, Union

import atexit
import logging
import os
import signal
import torch

from src.util import request_util
from src.util.debug_trace import debug_log
from src.data import data_type
import controller

from submodules.model.business_objects import general
from submodules.model import session, telemetry


OTLP_GRPC_ENDPOINT = os.getenv("OTLP_GRPC_ENDPOINT", "tempo:4317")

app_name = "refinery-embedder"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # region agent log
    debug_log(
        "app.py:lifespan",
        "application startup",
        {"pid": os.getpid()},
        "H6",
    )
    # endregion
    yield
    # region agent log
    debug_log(
        "app.py:lifespan",
        "application shutdown (uvicorn graceful stop)",
        {"pid": os.getpid()},
        "H6",
    )
    # endregion


app = FastAPI(title=app_name, lifespan=lifespan)


def _on_shutdown_signal(signum: int, _frame) -> None:
    # region agent log
    debug_log(
        "app.py:_on_shutdown_signal",
        "process received shutdown signal",
        {"signum": signum, "signal_name": signal.Signals(signum).name},
        "H1",
    )
    # endregion


def _on_process_exit() -> None:
    # region agent log
    debug_log(
        "app.py:_on_process_exit",
        "process exiting via atexit",
        {"pid": os.getpid()},
        "H1",
    )
    # endregion


signal.signal(signal.SIGTERM, _on_shutdown_signal)
signal.signal(signal.SIGINT, _on_shutdown_signal)
atexit.register(_on_process_exit)

def _cache_dir_writable(path: str | None) -> bool | None:
    if not path:
        return None
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".debug_b6799e_write_test")
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_file)
        return True
    except OSError:
        return False


# region agent log
debug_log(
    "app.py:startup",
    "refinery-embedder app module loaded",
    {
        "pid": os.getpid(),
        "hf_home": os.getenv("HF_HOME"),
        "hf_home_writable": _cache_dir_writable(os.getenv("HF_HOME")),
        "transformers_cache": os.getenv("TRANSFORMERS_CACHE"),
        "transformers_cache_writable": _cache_dir_writable(
            os.getenv("TRANSFORMERS_CACHE")
        ),
        "sentence_transformers_home": os.getenv("SENTENCE_TRANSFORMERS_HOME"),
        "sentence_transformers_home_writable": _cache_dir_writable(
            os.getenv("SENTENCE_TRANSFORMERS_HOME")
        ),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    },
    "H5",
)
# endregion

if telemetry.ENABLE_TELEMETRY:
    print("WARNING:  Running telemetry.", flush=True)
    telemetry.setting_app_name(app_name)
    telemetry.setting_otlp(app, app_name=app_name, endpoint=OTLP_GRPC_ENDPOINT)
    app.add_middleware(telemetry.PrometheusMiddleware, app_name=app_name)
    app.add_route("/metrics", telemetry.metrics)

    # Filter out /metrics
    logging.getLogger("uvicorn.access").addFilter(
        lambda record: "GET /metrics" not in record.getMessage()
    )


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
    # region agent log
    debug_log(
        "app.py:embed",
        "POST /embed received",
        {"project_id": request.project_id, "embedding_id": request.embedding_id},
        "H2",
    )
    # endregion
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
