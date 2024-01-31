import asyncio
import functools
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Type

import click
import pydantic
import yaml
from pydantic import ValidationError

from fedless.common.auth import CognitoClient


logger = logging.getLogger(__name__)


def get_available_cores():
    if hasattr(os, "sched_getaffinity"):
        num_available_cores = len(os.sched_getaffinity(0))
    else:
        logger.debug(
            "Method os.sched_getaffinity(0) not available, using os.cpu_count() instead "
            "which can give wrong results. "
        )
        num_available_cores = os.cpu_count()
    return num_available_cores


MAX_THREAD_POOL_WORKERS = 500
logger.info(
    f"Thread pool worker maximum set to {MAX_THREAD_POOL_WORKERS}, {get_available_cores()} available cores found"
)
_pool = ThreadPoolExecutor(max_workers=MAX_THREAD_POOL_WORKERS)


def parse_yaml_file(path, model: Optional[Type[pydantic.BaseModel]] = None):
    with open(path) as f:
        file_dict = yaml.safe_load(f)
    if not model:
        return file_dict
    try:
        return model.parse_obj(file_dict)
    except (KeyError, ValidationError) as e:
        raise click.ClickException(str(e))


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(_pool, lambda: f(*args, **kwargs))

    return inner


def fetch_cognito_auth_token(
    user_pool_id,
    region_name,
    auth_endpoint,
    invoker_client_id,
    invoker_client_secret,
    required_scopes,
) -> str:
    cognito = CognitoClient(
        user_pool_id=user_pool_id,
        region_name=region_name,
    )
    return cognito.fetch_token_for_client(
        auth_endpoint=auth_endpoint,
        client_id=invoker_client_id,
        client_secret=invoker_client_secret,
        required_scopes=required_scopes,
    )
