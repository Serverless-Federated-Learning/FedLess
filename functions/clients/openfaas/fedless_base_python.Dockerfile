FROM --platform=${TARGETPLATFORM:-linux/amd64} python:3.7-slim-buster

RUN apt-get update && apt-get install -y git
COPY fedless_requirements.txt .
USER root
RUN pip install --no-cache-dir -r fedless_requirements.txt
