ARG PARENT_IMAGE=registry.dev.kern.ai/code-kern-ai/refinery-parent-images:parent-image-updates-torch-cpu
ARG DHI_PYTHON_BUILD=dhi.io/python:3.11-debian12-dev

FROM ${PARENT_IMAGE} AS venv-source

FROM ${DHI_PYTHON_BUILD} AS builder

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /program

COPY --from=venv-source ${VENV_PATH} ${VENV_PATH}

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir -p /inference "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}" && \
    chown -R 65532:65532 /inference "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}"

COPY . .

FROM ${PARENT_IMAGE}

ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /program

COPY --from=builder --chown=65532:65532 ${VENV_PATH} ${VENV_PATH}
COPY --from=builder --chown=65532:65532 /inference /inference
COPY --from=builder --chown=65532:65532 /tmp/huggingface /tmp/huggingface
COPY --from=builder --chown=65532:65532 /tmp/sentence-transformers /tmp/sentence-transformers
COPY --from=builder --chown=65532:65532 /program /program

USER nonroot

CMD ["/opt/venv/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app"]
