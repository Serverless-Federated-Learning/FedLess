import logging
import os
from urllib import request
from pydantic import ValidationError
from flask import Flask, request

app = Flask(__name__)

from fedless.client import (
    fedless_mongodb_handler,
    ClientError,
)
from fedless.common.models import InvokerParams
from fedless.common.providers import openfaas_action_handler

logging.basicConfig(level=logging.DEBUG)


@openfaas_action_handler(caught_exceptions=(ValidationError, ClientError))
def handle(request):
    body: bytes = request.get_data()
    config = InvokerParams.parse_raw(body)

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
    )


@app.route("/", methods=["POST"])
def hello_world():
    logging.info(f"Request Recieved")
    content_type = request.headers.get("Content-Type")
    logging.info(f"Request [{content_type}]")
    if content_type == "application/json":
        json = request.get_json()
        logging.info(f"Request [{request}]")
        temp = handle(request)
        logging.info(f"{temp}")
        return temp
    else:
        return "Content-Type not supported!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
