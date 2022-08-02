# -*- coding: utf-8 -*-
from submodules.model import enums
from submodules.model.business_objects import (
    attribute,
    embedding,
    general,
    project,
    record,
    tokenization,
    notification,
    organization,
)
import torch
import traceback
from requests.exceptions import HTTPError
import logging
import time
import zlib
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from data import data_type, doc_ock
from embedders import Transformer
from typing import Any, Dict, Iterator, List

from util import daemon, request_util
from util.decorator import param_throttle
from util.embedders import get_embedder
from util.notification import send_project_update
import os
import pandas as pd
from submodules.s3 import controller as s3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_batches(
    project_id: str,
    record_ids: List[str],
    embedding_type: str,
    attribute_values_raw: List[str],
    embedder: Transformer,
    attribute_name: str,
) -> Iterator[Dict[List[str], List[Any]]]:
    length = len(record_ids)
    record_batches = []
    document_batches = []
    for idx in range(0, length, embedder.batch_size):
        record_ids_batch = record_ids[idx : min(idx + embedder.batch_size, length)]
        if embedding_type == "classification":
            documents = attribute_values_raw[
                idx : min(idx + embedder.batch_size, length)
            ]
        else:
            documents = get_docbins(
                project_id, record_ids_batch, embedder.nlp.vocab, attribute_name
            )
        record_batches.append(record_ids_batch)
        document_batches.extend(documents)

    embedding_batches = embedder.fit_transform(document_batches, as_generator=True)
    for record_batch in record_batches:
        yield {"record_ids": record_batch, "embeddings": next(embedding_batches)}


def get_docbins(
    project_id: str, record_ids_batch: List[str], vocab: Vocab, attribute_name: str
) -> List[Doc]:
    tokenized_records = tokenization.get_records_tokenized(project_id, record_ids_batch)
    result = {}
    for record_item in tokenized_records:
        doc_bin_loaded = DocBin().from_bytes(record_item.bytes)
        docs = list(doc_bin_loaded.get_docs(vocab))
        for (col, doc) in zip(record_item.columns, docs):
            if col == attribute_name:
                result[record_item.record_id] = doc
    result_list = []
    for record_id in record_ids_batch:
        result_list.append(result[record_id])
    return result_list


def start_encoding_thread(request: data_type.Request, embedding_type: str) -> int:
    doc_ock.post_embedding_creation(request.user_id, request.config_string)
    daemon.run(prepare_run_encoding, request, embedding_type)
    return 200


def prepare_run_encoding(request: data_type.Request, embedding_type: str) -> int:

    session_token = general.get_ctx_token()
    attribute_item = attribute.get(request.project_id, request.attribute_id)
    attribute_name = attribute_item.name
    attribute_data_type = attribute_item.data_type
    embedding_item = embedding.create(
        request.project_id,
        f"{attribute_name}-{embedding_type}-{request.config_string}",
        type=__infer_enum_value(embedding_type),
        with_commit=True,
    )
    embedding_id = str(embedding_item.id)
    send_project_update(
        request.project_id,
        f"embedding:{embedding_id}:state:{enums.EmbeddingState.INITIALIZING.value}",
    )

    if embedding_type == "extraction":

        progress = tokenization.get_doc_bin_progress(request.project_id)
        if progress or progress == 0:
            embedding.update_embedding_state_waiting(request.project_id, embedding_id)
            send_project_update(
                request.project_id,
                f"embedding:{embedding_id}:state:{enums.EmbeddingState.WAITING.value}",
            )
            counter = 0
            while progress or progress == 0:
                time.sleep(30)
                progress = tokenization.get_doc_bin_progress(request.project_id)
                counter += 1
                if counter >= 40:
                    embedding.update_embedding_state_failed(
                        request.project_id,
                        embedding_id,
                        with_commit=True,
                    )
                    send_project_update(
                        request.project_id,
                        f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
                    )
                    message = "Tokenization still in progress, aborting embedding creation. Please contact the support or retry later."
                    notification.create(
                        request.project_id,
                        request.user_id,
                        message,
                        "ERROR",
                        enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
                        True,
                    )
                    send_project_update(
                        request.project_id,
                        f"notification_created:{request.user_id}",
                        True,
                    )
                    embedding.update_embedding_state_failed(
                        request.project_id,
                        embedding_id,
                        with_commit=True,
                    )
                    doc_ock.post_embedding_failed(
                        request.user_id, request.config_string
                    )
                    raise Exception(message)
    general.remove_and_refresh_session(session_token)
    return run_encoding(
        request, embedding_id, embedding_type, attribute_name, attribute_data_type
    )


def run_encoding(
    request: data_type.Request,
    embedding_id: str,
    embedding_type: str,
    attribute_name: str,
    attribute_data_type: str,
) -> int:
    session_token = general.get_ctx_token()
    initial_count = record.count(request.project_id)
    seed_str = f"{attribute_name}-{attribute_data_type}-{request.config_string}-{embedding_type}"

    torch.manual_seed(zlib.adler32(bytes(seed_str, "utf-8")))
    try:

        notification.create(
            request.project_id,
            request.user_id,
            f"Initializing model {request.config_string}. This can take a few minutes.",
            "INFO",
            enums.NotificationType.EMBEDDING_CREATION_STARTED.value,
            True,
        )
        send_project_update(
            request.project_id, f"notification_created:{request.user_id}", True
        )
        iso2_code = project.get_blank_tokenizer_from_project(request.project_id)
        try:
            embedder = get_embedder(
                request.project_id, embedding_type, request.config_string, iso2_code
            )
        except OSError:
            embedding.update_embedding_state_failed(
                request.project_id,
                embedding_id,
                with_commit=True,
            )
            send_project_update(
                request.project_id,
                f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
            )
            doc_ock.post_embedding_failed(request.user_id, request.config_string)
            message = f"Model {request.config_string} is not supported. Please contact the support."
            notification.create(
                request.project_id,
                request.user_id,
                message,
                "ERROR",
                enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
                True,
            )
            send_project_update(
                request.project_id, f"notification_created:{request.user_id}", True
            )
            return 422

        if not embedder:
            embedding.update_embedding_state_failed(
                request.project_id,
                embedding_id,
                with_commit=True,
            )
            send_project_update(
                request.project_id,
                f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
            )
            doc_ock.post_embedding_failed(request.user_id, request.config_string)
            message = f"The data type {attribute_data_type} is currently not supported for embeddings. Please contact the support."
            notification.create(
                request.project_id,
                request.user_id,
                message,
                "ERROR",
                enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
                True,
            )
            send_project_update(
                request.project_id, f"notification_created:{request.user_id}", True
            )
            raise Exception(message)
    except HTTPError:
        print(traceback.format_exc(), flush=True)
        notification.create(
            request.project_id,
            request.user_id,
            f"Could not load model {request.config_string}. Please contact the support.",
            "ERROR",
            enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
            True,
        )
        send_project_update(
            request.project_id, f"notification_created:{request.user_id}", True
        )
        embedding.update_embedding_state_failed(
            request.project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            request.project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
        )
        doc_ock.post_embedding_failed(request.user_id, request.config_string)
        return 422

    try:
        record_ids, attribute_values_raw = record.get_attribute_data(
            request.project_id, attribute_name
        )
        embedding.update_embedding_state_encoding(
            request.project_id,
            embedding_id,
            with_commit=True,
        )
        send_progress_update_throttle(
            request.project_id,
            embedding_id,
            enums.EmbeddingState.ENCODING.value,
            initial_count,
        )
        send_project_update(
            request.project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.ENCODING.value}",
        )
        doc_ock.post_embedding_encoding(request.user_id, request.config_string)
        notification.create(
            request.project_id,
            request.user_id,
            f"Started encoding {attribute_name} using model {request.config_string}.",
            "INFO",
            enums.NotificationType.EMBEDDING_CREATION_STARTED.value,
            True,
        )
        send_project_update(
            request.project_id, f"notification_created:{request.user_id}", True
        )
        embedding.delete_tensors(embedding_id, with_commit=True)
        chunk = 0
        for pair in generate_batches(
            request.project_id,
            record_ids,
            embedding_type,
            attribute_values_raw,
            embedder,
            attribute_name,
        ):
            if chunk % 10 == 0:
                session_token = general.remove_and_refresh_session(session_token, True)

            record_ids_batched = pair["record_ids"]
            attribute_values_encoded_batch = pair["embeddings"]
            if not embedding.get(request.project_id, embedding_id):
                logger.info(
                    f"Aborted {attribute_name}-{embedding_type}-{request.config_string}"
                )
                break
            embedding.create_tensors(
                request.project_id,
                embedding_id,
                record_ids_batched,
                attribute_values_encoded_batch,
                with_commit=True,
            )
            send_progress_update_throttle(
                request.project_id,
                embedding_id,
                enums.EmbeddingState.ENCODING.value,
                initial_count,
            )
    except Exception:
        embedding.update_embedding_state_failed(
            request.project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            request.project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
        )
        notification.create(
            request.project_id,
            request.user_id,
            "Error at runtime. Please contact support.",
            "ERROR",
            enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
            True,
        )
        send_project_update(
            request.project_id, f"notification_created:{request.user_id}", True
        )
        print(traceback.format_exc(), flush=True)
        embedding.update_embedding_state_failed(
            request.project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            request.project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
        )
        doc_ock.post_embedding_failed(request.user_id, request.config_string)
        return 500

    if embedding.get(request.project_id, embedding_id):
        if embedding_type == "classification":
            request_util.post_embedding_to_neural_search(
                request.project_id, embedding_id
            )
        upload_embedding_as_file(request.project_id, embedding_id)
        embedding.update_embedding_state_finished(
            request.project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            request.project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FINISHED.value}",
        )
        notification_msg = (
            f"Finished encoding {attribute_name} using model {request.config_string}."
        )
        if len(embedder.get_warnings()):
            notification_msg += " Warnings: " + embedder.get_warnings()

        notification.create(
            request.project_id,
            request.user_id,
            notification_msg,
            "SUCCESS",
            enums.NotificationType.EMBEDDING_CREATION_DONE.value,
            True,
        )
        send_project_update(
            request.project_id, f"notification_created:{request.user_id}", True
        )
        doc_ock.post_embedding_finished(request.user_id, request.config_string)
    general.commit()
    general.remove_and_refresh_session(session_token)
    return 200


def delete_embedding(project_id: str, embedding_id: str) -> int:
    embedding.delete(project_id, embedding_id)
    embedding.delete_tensors(embedding_id)
    general.commit()
    object_name = f"embedding_tensors_{embedding_id}.csv.bz2"

    org_id = organization.get_id_by_project_id(project_id)
    s3.delete_object(org_id, project_id + "/" + object_name)
    request_util.delete_embedding_from_neural_search(embedding_id)
    return 200


@param_throttle(seconds=5)
def send_progress_update_throttle(
    project_id: str, embedding_id: str, state: str, initial_count: int
) -> None:
    progress = resolve_progress(embedding_id, state, initial_count)
    send_project_update(project_id, f"embedding:{embedding_id}:progress:{progress}")


def resolve_progress(embedding_id: str, state: str, initial_count: int) -> float:
    progress = 0.1 if state != "INITIALIZING" else 0
    progress += embedding.get_tensor_count(embedding_id) / initial_count * 0.9

    return min(progress, 0.99)


def __infer_enum_value(embedding_type: str) -> str:
    if embedding_type == "classification":
        return enums.EmbeddingType.ON_ATTRIBUTE.value
    elif embedding_type == "extraction":
        return enums.EmbeddingType.ON_TOKEN.value


def upload_embedding_as_file(
    project_id: str, embedding_id: str, force_recreate: bool = True
) -> None:
    org_id = organization.get_id_by_project_id(project_id)
    if not embedding.get(project_id, embedding_id):
        raise ValueError(
            f"no matching embedding {embedding_id} in project {project_id}"
        )
    if not s3.bucket_exists(org_id):
        s3.create_bucket(org_id)

    file_name = f"embedding_tensors_{embedding_id}.csv.bz2"
    s3_file_name = project_id + "/" + file_name
    exists = s3.object_exists(org_id, s3_file_name)
    if force_recreate and exists:
        s3.delete_object(org_id, s3_file_name)
    elif exists:
        return
    query = embedding.get_tensor_data_ordered_query(embedding_id)
    if os.path.exists(file_name):
        os.remove(file_name)

    for sql_df in pd.read_sql(query, con=general.get_bind(), chunksize=100):
        sql_df.to_csv(file_name, mode="a", index=False)
    s3.upload_object(org_id, s3_file_name, file_name)
    os.remove(file_name)
