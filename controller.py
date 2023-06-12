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
from fastapi import status
import pickle
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
from typing import Any, Dict, Iterator, List, Optional

from util import daemon, request_util
from util.config_handler import get_config_value
from util.decorator import param_throttle
from util.embedders import get_embedder
from util.notification import send_project_update, embedding_warning_templates
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
        if embedding_type == enums.EmbeddingType.ON_ATTRIBUTE.value:
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


def manage_encoding_thread(project_id: str, embedding_id: str) -> int:
    daemon.run(prepare_run_encoding, project_id, embedding_id)
    return status.HTTP_200_OK


def prepare_run_encoding(project_id: str, embedding_id: str) -> int:
    session_token = general.get_ctx_token()
    embedding_item = embedding.get(project_id, embedding_id)
    attribute_item = attribute.get(project_id, embedding_item.attribute_id)
    attribute_name = attribute_item.name
    attribute_data_type = attribute_item.data_type
    platform = embedding_item.platform
    embedding_id = str(embedding_item.id)
    user_id = embedding_item.created_by
    embedding_type=embedding_item.type
    model=embedding_item.model
    api_token=embedding_item.api_token
    embedding_name=embedding_item.name
    send_project_update(
        project_id,
        f"embedding:{embedding_id}:state:{enums.EmbeddingState.INITIALIZING.value}",
    )
    if embedding_type == enums.EmbeddingType.ON_TOKEN.value:

        progress = tokenization.get_doc_bin_progress(project_id)
        if progress or progress == 0:
            embedding.update_embedding_state_waiting(project_id, embedding_id)
            send_project_update(
                project_id,
                f"embedding:{embedding_id}:state:{enums.EmbeddingState.WAITING.value}",
            )
            counter = 0
            while progress or progress == 0:
                time.sleep(30)
                progress = tokenization.get_doc_bin_progress(project_id)
                counter += 1
                if counter >= 40:
                    embedding.update_embedding_state_failed(
                        project_id,
                        embedding_id,
                        with_commit=True,
                    )
                    send_project_update(
                        project_id,
                        f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
                    )
                    message = "Tokenization still in progress, aborting embedding creation. Please contact the support or retry later."
                    notification.create(
                        project_id,
                        user_id,
                        message,
                        enums.Notification.ERROR.value,
                        enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
                        True,
                    )
                    send_project_update(
                        project_id,
                        f"notification_created:{user_id}",
                        True,
                    )
                    doc_ock.post_embedding_failed(
                        user_id, model
                    )
                    raise Exception(message)
    general.remove_and_refresh_session(session_token)
    return run_encoding(
        project_id, user_id, embedding_id, embedding_type, embedding_name, attribute_name, attribute_data_type, platform, model, api_token
    )


def run_encoding(
    project_id: str,
    user_id: str,
    embedding_id: str,
    embedding_type: str,
    embedding_name:str,
    attribute_name: str,
    attribute_data_type: str,
    platform: str,
    model: Optional[str] = None,
    api_token: Optional[str] = None,
) -> int:
    session_token = general.get_ctx_token()
    initial_count = record.count(project_id)
    seed_str = embedding_name
    torch.manual_seed(zlib.adler32(bytes(seed_str, "utf-8")))
    notification.create(
        project_id,
        user_id,
        f"Initializing model {model}. This can take a few minutes.",
        enums.Notification.INFO.value,
        enums.NotificationType.EMBEDDING_CREATION_STARTED.value,
        True,
    )
    send_project_update(
        project_id, f"notification_created:{user_id}", True
    )
    iso2_code = project.get_blank_tokenizer_from_project(project_id)
    try:
        if platform == "huggingface":
            if not __is_embedders_internal_model(
                model
            ) and get_config_value("is_managed"):
                config_string = request_util.get_model_path(model)
                if type(config_string) == dict:
                    config_string = model
        else:
            config_string = model

        embedder = get_embedder(
            project_id, embedding_type, iso2_code, platform, model, api_token
        )

        if not embedder:
            raise Exception(f"The data type {attribute_data_type} is currently not supported for embeddings. Please contact the support.")
    except Exception as e:
        embedding.update_embedding_state_failed(
            project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
        )
        doc_ock.post_embedding_failed(user_id, model)
        message = f"Error while getting model - {e}"
        notification.create(
            project_id,
            user_id,
            message,
            enums.Notification.ERROR.value,
            enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
            True,
        )
        send_project_update(
            project_id, f"notification_created:{user_id}", True
        )
        return status.HTTP_422_UNPROCESSABLE_ENTITY

    try:
        record_ids, attribute_values_raw = record.get_attribute_data(
            project_id, attribute_name
        )
        embedding.update_embedding_state_encoding(
            project_id,
            embedding_id,
            with_commit=True,
        )
        send_progress_update_throttle(
            project_id,
            embedding_id,
            enums.EmbeddingState.ENCODING.value,
            initial_count,
        )
        send_project_update(
            project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.ENCODING.value}",
        )
        doc_ock.post_embedding_encoding(user_id, model)
        notification.create(
            project_id,
            user_id,
            f"Started encoding {attribute_name} using model {model}.",
            enums.Notification.INFO.value,
            enums.NotificationType.EMBEDDING_CREATION_STARTED.value,
            True,
        )
        send_project_update(
            project_id, f"notification_created:{user_id}", True
        )
        embedding.delete_tensors(embedding_id, with_commit=True)
        chunk = 0
        for pair in generate_batches(
            project_id,
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
            if not embedding.get(project_id, embedding_id):
                logger.info(
                    f"Aborted {attribute_name}-{embedding_type}-{model}"
                )
                break
            embedding.create_tensors(
                project_id,
                embedding_id,
                record_ids_batched,
                attribute_values_encoded_batch,
                with_commit=True,
            )
            send_progress_update_throttle(
                project_id,
                embedding_id,
                enums.EmbeddingState.ENCODING.value,
                initial_count,
            )
    except Exception:
        for warning_type, idx_list in embedder.get_warnings().items():
            # use last record with warning as example
            example_record_id = record_ids[idx_list[-1]]

            primary_keys = [
                pk.name for pk in attribute.get_primary_keys(project_id)
            ]
            if primary_keys:
                example_record_data = record.get(
                    project_id, example_record_id
                ).data
                example_record_msg = "with primary key: " + ", ".join(
                    [str(example_record_data[p_key]) for p_key in primary_keys]
                )
            else:
                example_record_msg = " with record id: " + str(example_record_id)

            warning_msg = embedding_warning_templates[warning_type].format(
                record_number=len(idx_list), example_record_msg=example_record_msg
            )

            notification.create(
                project_id,
                user_id,
                warning_msg,
                enums.Notification.WARNING.value,
                enums.NotificationType.EMBEDDING_CREATION_WARNING.value,
                True,
            )
            send_project_update(
                project_id, f"notification_created:{user_id}", True
            )

        embedding.update_embedding_state_failed(
            project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FAILED.value}",
        )
        notification.create(
            project_id,
            user_id,
            "Error at runtime. Please contact support.",
            enums.Notification.ERROR.value,
            enums.NotificationType.EMBEDDING_CREATION_FAILED.value,
            True,
        )
        send_project_update(
            project_id, f"notification_created:{user_id}", True
        )
        print(traceback.format_exc(), flush=True)
        doc_ock.post_embedding_failed(user_id, model)
        return status.HTTP_500_INTERNAL_SERVER_ERROR

    if embedding.get(project_id, embedding_id):
        for warning_type, idx_list in embedder.get_warnings().items():
            # use last record with warning as example
            example_record_id = record_ids[idx_list[-1]]

            primary_keys = [
                pk.name for pk in attribute.get_primary_keys(project_id)
            ]
            if primary_keys:
                example_record_data = record.get(
                    project_id, example_record_id
                ).data
                example_record_msg = "with primary key: " + ", ".join(
                    [str(example_record_data[p_key]) for p_key in primary_keys]
                )
            else:
                example_record_msg = " with record id: " + str(example_record_id)

            warning_msg = embedding_warning_templates[warning_type].format(
                record_number=len(idx_list), example_record_msg=example_record_msg
            )

            notification.create(
                project_id,
                user_id,
                warning_msg,
                enums.Notification.WARNING.value,
                enums.NotificationType.EMBEDDING_CREATION_WARNING.value,
                True,
            )
            send_project_update(
                project_id, f"notification_created:{user_id}", True
            )

        if embedding_type == enums.EmbeddingType.ON_ATTRIBUTE.value:
            request_util.post_embedding_to_neural_search(
                project_id, embedding_id
            )

        if get_config_value("is_managed"):
            pickle_path = os.path.join(
                "/inference", project_id, f"embedder-{embedding_id}.pkl"
            )
            if not os.path.exists(pickle_path):
                os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
                with open(pickle_path, "wb") as f:
                    pickle.dump(embedder, f)

        upload_embedding_as_file(project_id, embedding_id)
        embedding.update_embedding_state_finished(
            project_id,
            embedding_id,
            with_commit=True,
        )
        send_project_update(
            project_id,
            f"embedding:{embedding_id}:state:{enums.EmbeddingState.FINISHED.value}",
        )
        notification.create(
            project_id,
            user_id,
            f"Finished encoding {attribute_name} using model {model}.",
            enums.Notification.SUCCESS.value,
            enums.NotificationType.EMBEDDING_CREATION_DONE.value,
            True,
        )
        send_project_update(
            project_id, f"notification_created:{user_id}", True
        )
        doc_ock.post_embedding_finished(user_id, model)
    general.commit()
    general.remove_and_refresh_session(session_token)
    return status.HTTP_200_OK


def delete_embedding(project_id: str, embedding_id: str) -> int:
    object_name = f"embedding_tensors_{embedding_id}.csv.bz2"
    org_id = organization.get_id_by_project_id(project_id)
    s3.delete_object(org_id, f"{project_id}/{object_name}")
    request_util.delete_embedding_from_neural_search(embedding_id)
    pickle_path = os.path.join("/inference", project_id, f"embedder-{embedding_id}.pkl")
    if os.path.exists(pickle_path):
        os.remove(pickle_path)
    return status.HTTP_200_OK


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


def __is_embedders_internal_model(model_name: str):
    return model_name in ["bag-of-characters", "bag-of-words", "tf-idf"]
