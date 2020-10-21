FROM nvcr.io/nvidia/pytorch:20.03-py3
COPY requirements.txt .
RUN apt-get update
RUN apt-get install -y rsync 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
WORKDIR /workspace
EXPOSE 8888
