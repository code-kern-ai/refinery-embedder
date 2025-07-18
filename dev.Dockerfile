FROM kernai/refinery-parent-images:parent-image-updates-torch-cpu

WORKDIR /app

VOLUME ["/app"]

COPY requirements*.txt .

RUN apt-get update && apt-get install -y git --no-install-recommends

RUN pip3 install --no-cache-dir -r requirements-dev.txt

COPY / .

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app", "--reload" ]