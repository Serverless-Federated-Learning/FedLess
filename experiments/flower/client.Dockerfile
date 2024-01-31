FROM mohamedazab/fedless:flower-custom

COPY ./client.py client.py

ENV TF_ENABLE_ONEDNN_OPTS=1
ENTRYPOINT ["python", "client.py"]
CMD ["--help"]