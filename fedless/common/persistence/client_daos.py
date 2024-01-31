from collections import defaultdict
from typing import Any, Iterator, List, Tuple, Union

import bson
import pymongo
from gridfs import GridFS
from gridfs.errors import GridFSError
from pymongo.errors import ConnectionFailure

from fedless.common.models import (
    ClientConfig,
    ClientPersistentHistory,
    ClientResult,
    MongodbConnectionConfig,
    SerializedModel,
    SerializedParameters,
)
from fedless.common.persistence.mongodb_base_connector import (
    DocumentAlreadyExistsException,
    DocumentNotLoadedException,
    MongoDbDao,
    PersistenceError,
    PersistenceValueError,
    StorageConnectionError,
    wrap_pymongo_errors,
)


class ClientResultDao(MongoDbDao):
    """Store client results in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "results",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        round_id: int,
        client_id: str,
        result: Union[dict, ClientResult],
        overwrite: bool = True,
    ) -> Any:
        if isinstance(result, ClientResult):
            result = result.dict()

        if (
            not overwrite
            and self._collection.find_one(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                }
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client result for session {session_id} and round {round_id} for client {client_id} already exists. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(result))
            self._collection.replace_one(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                },
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                    "file_id": file_id,
                },
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def load(
        self,
        session_id: str,
        round_id: int,
        client_id: str,
    ) -> ClientResult:
        try:
            obj_dict = self._collection.find_one(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                    "client_id": client_id,
                },
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Client result for session {session_id} and round {round_id} for client {client_id} not found."
            )

        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Client result for session {session_id} and client {client_id}" f"for round {round_id} malformed."
            )
        results_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not results_file:
            raise DocumentNotLoadedException(
                f"GridFS file with results for session {session_id} and client {client_id} "
                f"and round {round_id} not found."
            )
        try:
            return ClientResult.parse_obj(bson.decode(results_file.read()))
        finally:
            results_file.close()

    def _retrieve_result_files(self, result_dicts, session_id: str, round_id) -> Iterator[ClientResult]:
        for result_dict in result_dicts:
            if not result_dict:
                raise DocumentNotLoadedException(f"Client results for session {session_id} found.")
            client_round_id = result_dict["round_id"]
            if "file_id" not in result_dict:
                raise PersistenceValueError(
                    f"Client result in session {session_id},{round_id} for client_round {client_round_id} malformed."
                )
            results_file = self._gridfs.find_one({"_id": result_dict["file_id"]})
            if not results_file:
                raise DocumentNotLoadedException(
                    f"GridFS file with results in session {session_id},{round_id} "
                    f"and client round {client_round_id} not found."
                )
            try:
                yield ClientResult.parse_obj(bson.decode(results_file.read()))
            finally:
                results_file.close()

    @wrap_pymongo_errors
    def load_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ) -> Tuple[List, Iterator[ClientResult]]:
        try:
            result_dicts = list(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        files_iter = self._retrieve_result_files(result_dicts, session_id, round_id)
        return result_dicts, files_iter

    @wrap_pymongo_errors
    def load_client_id_in_round(self, session_id: str, round_id: int) -> List[str]:

        try:
            obj_dict = self._collection.find(
                filter={"session_id": session_id, "round_id": round_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Clients for session {session_id} and round {round_id} not found")
        clients_in_round = []
        for dict in obj_dict:
            clients_in_round.append(dict["client_id"])

        return clients_in_round

    @wrap_pymongo_errors
    def load_results_for_client_round(
        self, session_id: str, round_id: int, client_ids: List[str]
    ) -> Tuple[List, Iterator[ClientResult]]:
        try:
            result_dicts = list(
                self._collection.find(
                    filter={"session_id": session_id, "round_id": round_id, "client_id": {"$in": client_ids}},
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        files_iter = self._retrieve_result_files(result_dicts, session_id, round_id)
        return result_dicts, files_iter

    @wrap_pymongo_errors
    def load_results_for_session(
        self, session_id: str, round_id: int, tolerance: int
    ) -> Tuple[List, Iterator[ClientResult]]:
        try:
            result_dicts = list(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": {"$gte": round_id - tolerance},
                    }
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        files_iter = self._retrieve_result_files(result_dicts, session_id, round_id)
        return result_dicts, files_iter

    @wrap_pymongo_errors
    def delete_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def delete_results_for_session(
        self,
        session_id: str,
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                filter={
                    "session_id": session_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def count_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ) -> int:
        try:
            return self._collection.count_documents(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                },
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def count_results_for_session(
        self,
        session_id: str,
    ) -> int:
        try:
            return self._collection.count_documents(
                filter={
                    "session_id": session_id,
                },
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e


class ClientHistoryDao(MongoDbDao):
    """Store client history in the db for intelligent selection"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_history",
        database: str = "fedless",
    ):
        super().__init__(db, collection, database=database)

    @wrap_pymongo_errors
    def save(self, client: ClientPersistentHistory, overwrite: bool = True) -> Any:
        if not overwrite and self._collection.find_one({"client_id": client.client_id}) is not None:
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one({"client_id": client.client_id}, client.dict(), upsert=True)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, client_id: str) -> ClientPersistentHistory:
        try:
            obj_dict = self._collection.find_one(filter={"client_id": client_id})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Client with id {client_id} not found")
        return ClientPersistentHistory.parse_obj(obj_dict)

    @wrap_pymongo_errors
    def load_all(self, session_id: str) -> Iterator[ClientPersistentHistory]:
        try:
            obj_dicts = iter(self._collection.find(filter={"session_id": session_id}))
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for client_dict in obj_dicts:
            yield ClientPersistentHistory.parse_obj(client_dict)


class ClientConfigDao(MongoDbDao):
    """Store client configs in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "clients",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(self, client: ClientConfig, overwrite: bool = True) -> Any:
        if not overwrite and self._collection.find_one({"client_id": client.client_id}) is not None:
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one({"client_id": client.client_id}, client.dict(), upsert=True)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, client_id: str) -> ClientConfig:
        try:
            obj_dict = self._collection.find_one(filter={"client_id": client_id})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Client with id {client_id} not found")
        return ClientConfig.parse_obj(obj_dict)

    @wrap_pymongo_errors
    def load_all(self, session_id: str) -> Iterator[ClientConfig]:
        try:
            obj_dicts = iter(self._collection.find(filter={"session_id": session_id}))
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for client_dict in obj_dicts:
            yield ClientConfig.parse_obj(client_dict)

    @wrap_pymongo_errors
    def load_for_clients(self, session_id: str, client_ids: List[str]) -> Tuple[List, Iterator[ClientResult]]:
        try:
            obj_dicts = iter(
                self._collection.find(
                    filter={"session_id": session_id, "client_id": {"$in": client_ids}},
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

        for client_dict in obj_dicts:
            yield ClientConfig.parse_obj(client_dict)


class ParameterDao(MongoDbDao):
    """Store global model parameters in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "parameters",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        round_id: int,
        params: SerializedParameters,
        overwrite: bool = True,
    ) -> Any:

        if not overwrite and self._collection.find_one({"session_id": session_id, "round_id": round_id}) is not None:
            raise DocumentAlreadyExistsException(
                f"Parameters for session {session_id} and round {round_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(params.dict()), encoding="utf-8")
            self._collection.replace_one(
                {"session_id": session_id, "round_id": round_id},
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "file_id": file_id,
                },
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(
        self,
        session_id: str,
        round_id: int,
    ) -> SerializedParameters:
        try:
            obj_dict = self._collection.find_one(
                filter={"session_id": session_id, "round_id": round_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} and round {round_id} not found")
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} " f"and round {round_id} malformed."
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} and round {round_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def load_latest(self, session_id: str) -> SerializedParameters:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} not found")
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} not found")
        if "file_id" not in obj_dict:
            raise PersistenceValueError(f"Loaded parameters for session {session_id} " f"Expected key file_id")
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(f"GridFS file with parameters for session {session_id} not found")
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def get_latest_round(self, session_id: str) -> int:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} not found")
        if obj_dict is None or "round_id" not in obj_dict:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} not found or malformed")
        return int(obj_dict["round_id"])

    @wrap_pymongo_errors
    def delete_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e


class ModelDao(MongoDbDao):
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "models",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(self, session_id: str, model: SerializedModel, overwrite: bool = True) -> Any:
        if not overwrite and self._collection.find_one({"session_id": session_id}) is not None:
            raise DocumentAlreadyExistsException(
                f"Model architecture for session {session_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"session_id": session_id},
                {"session_id": session_id, "model": model.dict()},
                upsert=True,
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, session_id: str) -> SerializedModel:
        try:
            obj_dict = self._collection.find_one(filter={"session_id": session_id})
            obj_dict = obj_dict["model"] if obj_dict is not None and "model" in obj_dict else None

            if obj_dict is None:
                raise DocumentNotLoadedException(f"Client with id {id} not found")

            return SerializedModel.parse_obj(obj_dict)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except KeyError:
            raise PersistenceValueError(
                f"Loaded model architecture for session {session_id} malformed. Expected key parameters"
            )


class ClientModelDao(MongoDbDao):
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_models",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        client_id: str,
        model: SerializedModel,
        # logit_model: Optional[SerializedModel] = None,
        overwrite: bool = True,
    ) -> Any:
        if not overwrite and self._collection.find_one({"session_id": session_id, "client_id": client_id}) is not None:
            raise DocumentAlreadyExistsException(
                f"Model architecture for session {session_id} and client {client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"session_id": session_id, "client_id": client_id},
                {"session_id": session_id, "client_id": client_id, "model": model.dict()},
                upsert=True,
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def get_model_client_mapping(self, session_id: str):
        try:
            obj_dict = self._collection.find(filter={"session_id": session_id})

            if obj_dict is None:
                raise DocumentNotLoadedException(f"Sessions with id {session_id} not found")

            model_client_mapping = defaultdict(list)
            for dict in obj_dict:
                model_client_mapping[dict["model"]["model_type"]].append(dict["client_id"])

            return model_client_mapping

        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except KeyError:
            raise PersistenceValueError(
                f"Loaded client model architecture mapping for session {session_id} malformed. Expected key parameters"
            )

    @wrap_pymongo_errors
    def load(self, session_id: str, client_id: str) -> SerializedModel:
        try:
            obj_dict = self._collection.find_one(filter={"session_id": session_id, "client_id": client_id})
            obj_dict = obj_dict["model"] if obj_dict is not None and "model" in obj_dict else None

            if obj_dict is None:
                raise DocumentNotLoadedException(f"Client with id {id} not found")

            return SerializedModel.parse_obj(obj_dict)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except KeyError:
            raise PersistenceValueError(
                f"Loaded model architecture for session {session_id} and client {client_id} malformed. Expected key parameters"
            )


class ClientLogitPredictionsDao(MongoDbDao):
    """Store client logit predictions on public alignment data in a mongodb database (Specific to FedMD)"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_logits",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        client_id: str,
        round_id: int,
        logits: SerializedParameters,
        overwrite: bool = True,
    ) -> Any:

        if (
            not overwrite
            and self._collection.find_one({"session_id": session_id, "round_id": round_id, "client_id": client_id})
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Parameters for session {session_id} and round {round_id} and client {client_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(logits.dict()), encoding="utf-8")
            self._collection.replace_one(
                {"session_id": session_id, "round_id": round_id, "client_id": client_id},
                {"session_id": session_id, "round_id": round_id, "file_id": file_id, "client_id": client_id},
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, session_id: str, round_id: int, client_id: str) -> SerializedParameters:
        try:
            obj_dict = self._collection.find_one(
                filter={"session_id": session_id, "round_id": round_id, "client_id": client_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Logits for session {session_id} and round {round_id} and client {client_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded logits for session {session_id} " f"and round {round_id} and client { client_id} malformed."
            )
        logit_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not logit_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} and round {round_id} and client {client_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(logit_file.read()))
        finally:
            logit_file.close()

    @wrap_pymongo_errors
    def load_latest(self, session_id: str, client_id: str) -> SerializedParameters:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id, "client_id": client_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Logits for session {session_id} and client {client_id} not found")
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Logits for session {session_id} and client {client_id} not found")
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded logits for session {session_id} and client {client_id}" f"Expected key file_id"
            )
        logit_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not logit_file:
            raise DocumentNotLoadedException(f"GridFS file with parameters for session {session_id} not found")
        try:
            return SerializedParameters.parse_obj(bson.decode(logit_file.read()))
        finally:
            logit_file.close()

    @wrap_pymongo_errors
    def get_latest_round(self, session_id: str, client_id: str) -> int:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id, "client_id": client_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Logits for session {session_id} and client {client_id} not found")
        if obj_dict is None or "round_id" not in obj_dict:
            raise DocumentNotLoadedException(
                f"Logits for session {session_id} and client {client_id} not found or malformed"
            )
        return int(obj_dict["round_id"])

    def _retrieve_result_files(self, result_dicts, session_id: str, round_id) -> Iterator[SerializedParameters]:
        for result_dict in result_dicts:
            if not result_dict:
                raise DocumentNotLoadedException(f"Client results for session {session_id} found.")
            client_round_id = result_dict["round_id"]
            if "file_id" not in result_dict:
                raise PersistenceValueError(
                    f"Client result in session {session_id},{round_id} for client_round {client_round_id} malformed."
                )
            results_file = self._gridfs.find_one({"_id": result_dict["file_id"]})
            if not results_file:
                raise DocumentNotLoadedException(
                    f"GridFS file with results in session {session_id},{round_id} "
                    f"and client round {client_round_id} not found."
                )
            try:
                yield SerializedParameters.parse_obj(bson.decode(results_file.read()))
            finally:
                results_file.close()

    @wrap_pymongo_errors
    def load_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ) -> Tuple[List, Iterator[SerializedParameters]]:
        try:
            result_dicts = list(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        files_iter = self._retrieve_result_files(result_dicts, session_id, round_id)
        return result_dicts, files_iter

    @wrap_pymongo_errors
    def delete_results_for_round(
        self,
        session_id: str,
        round_id: int,
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": session_id,
                        "round_id": round_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                filter={
                    "session_id": session_id,
                    "round_id": round_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e


class ClientParameterDao(MongoDbDao):
    """Store client model parameters(weights) in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_parameters",
        database: str = "fedless",
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )
        try:
            self._gridfs = GridFS(self._client[self.database])
        except TypeError as e:
            raise PersistenceError(e) from e

    @wrap_pymongo_errors
    def load_client_id_in_round(self, session_id: str, round_id: int) -> List[str]:

        try:
            obj_dict = self._collection.find(
                filter={"session_id": session_id, "round_id": round_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Clients for session {session_id} and round {round_id} not found")
        clients_in_round = []
        for dict in obj_dict:
            clients_in_round.append(dict["client_id"])

        return clients_in_round

    @wrap_pymongo_errors
    def save(
        self,
        session_id: str,
        client_id: str,
        round_id: int,
        params: SerializedParameters,
        overwrite: bool = True,
    ) -> Any:

        if (
            not overwrite
            and self._collection.find_one({"session_id": session_id, "round_id": round_id, "client_id": client_id})
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Parameters for session {session_id} and round {round_id} and client {client_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(params.dict()), encoding="utf-8")
            self._collection.replace_one(
                {"session_id": session_id, "round_id": round_id, "client_id": client_id},
                {"session_id": session_id, "round_id": round_id, "file_id": file_id, "client_id": client_id},
                upsert=True,
            )
        except (ConnectionFailure, GridFSError) as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, session_id: str, round_id: int, client_id: str) -> SerializedParameters:
        try:
            obj_dict = self._collection.find_one(
                filter={"session_id": session_id, "round_id": round_id, "client_id": client_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} and round {round_id} and client {client_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} "
                f"and round {round_id} and client { client_id} malformed."
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} and round {round_id} and client {client_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def load_latest(self, session_id: str, client_id: str) -> SerializedParameters:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id, "client_id": client_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} and client {client_id} not found")
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} and client {client_id} not found")
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} and client {client_id}" f"Expected key file_id"
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(f"GridFS file with parameters for session {session_id} not found")
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def get_latest_round(self, session_id: str, client_id: str) -> int:
        try:
            obj_dict = (
                self._collection.find(
                    filter={"session_id": session_id, "client_id": client_id},
                )
                .sort("round_id", direction=pymongo.DESCENDING)
                .next()
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except StopIteration:
            raise DocumentNotLoadedException(f"Parameters for session {session_id} and client {client_id} not found")
        if obj_dict is None or "round_id" not in obj_dict:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} and client {client_id} not found or malformed"
            )
        return int(obj_dict["round_id"])

    @wrap_pymongo_errors
    def delete_all_except(
        self,
        session_id: List[str],
    ):
        try:
            result_dicts = iter(
                self._collection.find(
                    filter={
                        "session_id": {"$nin": session_id},
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(filter={"session_id": {"$nin": session_id}})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
