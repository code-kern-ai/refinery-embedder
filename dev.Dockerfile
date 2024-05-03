FROM kernai/refinery-parent-images:v1.15.0-torch-cpu

WORKDIR /app

VOLUME ["/app"]

COPY requirements*.txt .

RUN pip3 install --no-cache-dir -r requirements-dev.txt

COPY / .

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app", "--reload" ]