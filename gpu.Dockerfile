FROM registry.dev.kern.ai/code-kern-ai/refinery-parent-images:parent-image-updates-torch-cuda

WORKDIR /program

COPY gpu-requirements.txt .

RUN python3.9 -m pip install --no-cache-dir -r gpu-requirements.txt

COPY / .

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app" ]