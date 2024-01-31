FROM mohamedazab/fedless:flower

COPY ./server.py server.py
ENTRYPOINT ["python", "server.py"]
CMD ["--help"]