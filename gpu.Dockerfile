ARG PARENT_IMAGE=kernai/refinery-parent-images:v2.6.0-torch-cuda

FROM ${PARENT_IMAGE} AS builder

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /program

USER root

RUN if [ ! -d "${VENV_PATH}" ]; then python -m venv "${VENV_PATH}"; fi

COPY gpu-requirements.txt .

RUN pip3 install --no-cache-dir -r gpu-requirements.txt

COPY . .

FROM ${PARENT_IMAGE}

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers

WORKDIR /program

USER root

RUN mkdir -p /inference "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}" && \
    chown -R 65532:65532 /inference "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}"

COPY --from=builder --chown=65532:65532 ${VENV_PATH} ${VENV_PATH}
COPY --from=builder --chown=65532:65532 /program /program

USER nonroot

CMD ["/opt/venv/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app"]
