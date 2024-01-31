from contextlib import AbstractContextManager
from typing import Callable, Union

import pymongo
from pymongo.collection import Collection
from pymongo.errors import BSONError, InvalidName, PyMongoError

from fedless.common.models import MongodbConnectionConfig


class PersistenceError(Exception):
    """Base exception for persistence errors"""


class PersistenceValueError(PermissionError):
    """Unexpected or invalid value encountered"""


class StorageConnectionError(PersistenceError):
    """Connection to storage resource (e.g. database) could not be established"""


class DocumentNotStoredException(PersistenceError):
    """Client result could not be stored"""


class DocumentAlreadyExistsException(PersistenceError):
    """A result for this client already exists"""


class DocumentNotLoadedException(PersistenceError):
    """Client result could not be loaded"""


def wrap_pymongo_errors(func: Callable) -> Callable:
    """Decorator to wrap all unhandled pymongo exceptions as persistence errors"""

    # noinspection PyMissingOrEmptyDocstring
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (PyMongoError, BSONError) as e:
            raise PersistenceError(e) from e

    return wrapped_function


class MongoDbDao(AbstractContextManager):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str,
        database: str = "fedless",
    ):
        """
        Connect to mongodb database
        :param db: mongodb url or config object of type :class:`MongodbConnectionConfig`
        :param database: database name
        :param collection: collection name
        """
        self.db = db
        self.database = database
        self.collection = collection

        if isinstance(db, str):
            self._client = pymongo.MongoClient(db)
        elif isinstance(db, MongodbConnectionConfig):
            # self._client = pymongo.MongoClient(
            #     host=db.host,
            #     port=db.port,
            #     username=db.username,
            #     password=db.password,
            # )
            # only use url to be able to use mongo serve for sharded dbs
            self._client = pymongo.MongoClient(db.url)
        else:
            self._client = db

        try:
            self._collection: Collection = self._client[database][collection]
        except InvalidName as e:
            raise StorageConnectionError(e) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._client.close()
