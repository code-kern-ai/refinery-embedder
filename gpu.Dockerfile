FROM nvidia/cuda:11.4.0-base-ubuntu20.04
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.9
RUN apt-get -y install python3-pip

WORKDIR /program

COPY requirements.txt .

RUN python3.9 -m pip install -r requirements.txt

COPY / .

CMD [ "/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app" ]
