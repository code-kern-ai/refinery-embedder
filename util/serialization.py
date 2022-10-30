import os
import json
import embedders
from util.embedders import check_reduce_via_pca

MODEL_FOLDER = "/models"

def make_dir(
    project_id, 
    attribute_id,
    embedding_id, 
    embdder_config_string, 
    embedder, 
    embedding_type, 
    n_components
):
    dir_path = os.path.join(MODEL_FOLDER, f"embedder--{embedding_id}")

    store_pca_reduced = check_reduce_via_pca(embedding_type, embedder.__class__, project_id, n_components)
    embedder_config = {
        "project_id": project_id,
        "attribute_id": attribute_id,
        "embedding_id": embedding_id,
        "embedding_type": embedding_type,
        "config_string": embdder_config_string,
        "store_pca_reduced": store_pca_reduced,
    }

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    if store_pca_reduced:
        embedder: embedders.PCAReducer = embedder
        embedder.store_pca_weights(os.path.join(dir_path, "pca_weights"))
        embedder_config["n_components"] = n_components


    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump(embedder_config, f)

