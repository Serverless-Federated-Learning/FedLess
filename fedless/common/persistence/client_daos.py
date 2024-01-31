from datetime import datetime
import time
from typing import Any, List, Tuple, Union, Iterator

import bson
import pymongo
from gridfs import GridFS
from gridfs.errors import GridFSError
from pymongo.errors import ConnectionFailure
from fedless.common.models.models import InvocationHistory, InvocationStatus

from fedless.common.persistence.mongodb_base_connector import (
    DocumentAlreadyExistsException,
    DocumentNotLoadedException,
    MongoDbDao,
    PersistenceError,
    PersistenceValueError,
    StorageConnectionError,
    wrap_pymongo_errors,
)
from fedless.common.models import (
    ClientPersistentHistory,
    ClientScore,
    ClientResult,
    MongodbConnectionConfig,
    ClientConfig,
    SerializedParameters,
    SerializedModel,
)

DATABASE = "fedless"


class ClientResultDao(MongoDbDao):
    """Store client results in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "results",
        database: str = DATABASE,
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
    def count_available_results(self, session_id: str, tolerance_round: int):
        # tolerance_round = current round - tolerance
        return self._collection.count_documents(
            {"session_id": session_id, "round_id": {"$gte": tolerance_round}}
        )

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
                f"Client result for session {session_id} and client {client_id}"
                f"for round {round_id} malformed."
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

    def counter():
        pass

    @wrap_pymongo_errors
    def load_controls(
        self,
        session_id: str,
        round_id: int,
        client_id: str,
    ) -> SerializedParameters:
        client_result = self.load(
            session_id=session_id, round_id=round_id, client_id=client_id
        )
        return client_result.local_controls

    def _retrieve_result_files(
        self, result_dicts, session_id: str, round_id
    ) -> Iterator[ClientResult]:
        for result_dict in result_dicts:
            if not result_dict:
                raise DocumentNotLoadedException(
                    f"Client results for session {session_id} found."
                )
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
    def count_results_by_file_ids(
        self, session_id: str, tolerance_round_id: int, file_ids: List[str]
    ):
        try:
            return self._collection.count_documents(
                {
                    "session_id": session_id,
                    "$or": [
                        {"round_id": {"$lt": tolerance_round_id}},
                        {"file_id": {"$in": file_ids}},
                    ],
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def delete_results_by_file_ids(
        self, session_id: str, tolerance_round_id: int, file_ids: List[str]
    ):
        try:
            # get file ids from old & processed results
            result_dicts = iter(
                self._collection.find(
                    {
                        "session_id": session_id,
                        "$or": [
                            {"round_id": {"$lt": tolerance_round_id}},
                            {"file_id": {"$in": file_ids}},
                        ],
                    }
                )
            )

            # delete old & processed results
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                {
                    "session_id": session_id,
                    "$or": [
                        {"round_id": {"$lt": tolerance_round_id}},
                        {"file_id": {"$in": file_ids}},
                    ],
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


class InvocationHistoryDao(MongoDbDao):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        # session: str,
        collection: str = "invocation_history",
        database: str = DATABASE,
    ):
        super().__init__(db, collection, database)
        # self.session = session

    @wrap_pymongo_errors
    def get_cold_start_clients(
        self,
        session_id: str,
        clients: List[ClientConfig],
        cool_off_time=10 * 60,
        exec_timeout=20 * 60,
    ):
        client_ids = set([client.client_id for client in clients])
        warm_client_ids = self._collection.find(
            {
                "session_id": session_id,
                "$or": [
                    {
                        # if still running check if execution timeout, and cool down
                        # invocation time + exec_timeout + cool_off_time < current time
                        # -> current time - exec_timeout - cool_off_time > invocation time
                        "status": InvocationStatus.running,
                        "invocation_time": {
                            "$gte": time.time() - exec_timeout - cool_off_time
                        },
                    },
                    {
                        # if error check inv timeout
                        "status": {"$lt": 0},
                        "invocation_time": {"$gte": time.time() - cool_off_time},
                    },
                    {
                        # if completed check if still warm
                        "status": InvocationStatus.completed,
                        "complete_time": {"$gte": time.time() - cool_off_time},
                    },
                ],
            }
        ).distinct("client_id")

        available_client_ids = client_ids - set(warm_client_ids)
        return list(available_client_ids)

    @wrap_pymongo_errors
    def get_running_count(self, session_id: str, tolerance_round: int):
        return self._collection.count_documents(
            {
                "session_id": session_id,
                "round_id": {"$gte": tolerance_round},
                "status": InvocationStatus.running,
            }
        )

    @wrap_pymongo_errors
    def get_last_failed_clients(self, session_id: str, round_id: int) -> List[str]:
        client_lists = list(
            self._collection.find(
                {
                    "session_id": session_id,
                    "round_id": {"$gte": round_id},
                    "status": {"$lt": 0},  # fail and timeout
                },
                {"client_id": 1, "_id": 0},
            )
        )

        return [client["client_id"] for client in client_lists]

    @wrap_pymongo_errors
    def get_distinct_inv_ids(self, session: str):
        return self._collection.find({"session_id": session}).distinct("client_id")

    @wrap_pymongo_errors
    def get_busy_inv_client_ids(self, session: str, exec_timeout: int = 20 * 60):
        # get busy clients that hasn't timeout yet
        return self._collection.find(
            {
                "session_id": session,
                "status": InvocationStatus.running,
                "invocation_time": {"$lte": time.time() - exec_timeout},
            }
        ).distinct("client_id")

    @wrap_pymongo_errors
    def save_invocation(
        self, invocation_history: InvocationHistory, overwrite: bool = True
    ) -> Any:
        if (
            not overwrite
            and self._collection.find_one(
                {"invocation_id": invocation_history.invocation_id}
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Invocation history with id {invocation_history.invocation_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"invocation_id": invocation_history.invocation_id},
                invocation_history.dict(),
                upsert=True,
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def save_batch_invocations(self, invocation_history: List[InvocationHistory]):
        try:
            self._collection.insert_many([hist.dict() for hist in invocation_history])
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def _update_one(self, invocation_id: str, updates: dict):
        try:
            self._collection.update_one(
                {"invocation_id": invocation_id}, {"$set": updates}
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def inv_done(
        self,
        invocation_id: str,
        invocation_time: float,
        complete_time: float,
        status: InvocationStatus = InvocationStatus.completed,
    ):
        self._update_one(
            invocation_id,
            {
                "invocation_time": invocation_time,
                "complete_time": complete_time,
                "response_time": complete_time - invocation_time,
                "status": status,
            },
        )

    @wrap_pymongo_errors
    def exec_done(self, invocation_id: str, function_duration: float):
        self._update_one(
            invocation_id,
            {
                "function_duration": function_duration,
                "status": InvocationStatus.completed,
            },
        )

    @wrap_pymongo_errors
    def inv_timeout(self, invocation_id: str):
        self._update_one(invocation_id, {"status": InvocationStatus.inv_timeout})

    @wrap_pymongo_errors
    def exec_timeout(self, invocation_id: str):
        self._update_one(invocation_id, {"status": InvocationStatus.exec_timeout})

    @wrap_pymongo_errors
    def inv_fail(self, invocation_id: str):
        # TODO: decrease booster value for penalty
        self._update_one(invocation_id, {"status": InvocationStatus.failed})


class ClientScoreDao(MongoDbDao):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_scores",
        database: str = DATABASE,
    ):
        super().__init__(db, collection, database)

    @wrap_pymongo_errors
    def reset_booster_value(self, ids: List[str]):
        self._collection.update_many(
            {"client_id": {"$in": ids}}, {"$set": {"booster_value": 1.0}}
        )

    @wrap_pymongo_errors
    def update_booster_value(self, ids: List[str], scale: float):
        self._collection.update_many(
            {"client_id": {"$in": ids}}, {"$mul": {"booster_value": scale}}
        )

    @wrap_pymongo_errors
    def update_stats(self, client_id: str, training_time: float) -> Any:
        if self._collection.find_one({"client_id": client_id}):
            try:
                self._collection.update_one(
                    {"client_id": client_id},
                    {"$push": {"training_times": training_time}},
                )
            except ConnectionFailure as e:
                raise StorageConnectionError(e) from e
        else:
            raise DocumentNotLoadedException(
                f"Client score stats for client {client_id} not found."
            )

    @wrap_pymongo_errors
    def stats_exists(self, client_id: str) -> bool:
        return self._collection.find_one({"client_id": client_id}) is not None

    @wrap_pymongo_errors
    def save(self, client: ClientScore, overwrite: bool = True) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"client_id": client.client_id}) is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"client_id": client.client_id}, client.dict(), upsert=True
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    @wrap_pymongo_errors
    def load(self, client_id: str) -> ClientScore:
        try:
            obj_dict = self._collection.find_one(filter={"client_id": client_id})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(f"Client with id {client_id} not found")
        return ClientScore.parse_obj(obj_dict)

    @wrap_pymongo_errors
    def full_load_all(self) -> Iterator[ClientScore]:
        try:
            obj_dicts = iter(self._collection.find())
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for client_dict in obj_dicts:
            yield ClientScore.parse_obj(client_dict)

    @wrap_pymongo_errors
    def load_all(self, n: int = 5) -> Iterator[ClientScore]:
        # only get last n training times
        try:
            obj_dicts = iter(
                self._collection.find({}, {"training_times": {"$slice": -n}})
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        for client_dict in obj_dicts:
            yield ClientScore.parse_obj(client_dict)

    @wrap_pymongo_errors
    def load_all_client_ids(self) -> Iterator[str]:
        try:
            obj_dicts = iter(self._collection.find({}, {"client_id": 1, "_id": 0}))
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        return [client_dict["client_id"] for client_dict in obj_dicts]


class ClientHistoryDao(MongoDbDao):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_history",
        database: str = DATABASE,
    ):
        super().__init__(db, collection, database=database)

    @wrap_pymongo_errors
    def save(self, client: ClientPersistentHistory, overwrite: bool = True) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"client_id": client.client_id}) is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one(
                {"client_id": client.client_id}, client.dict(), upsert=True
            )
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
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "clients",
        database: str = DATABASE,
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(self, client: ClientConfig, overwrite: bool = True) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"client_id": client.client_id}) is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Client with id {client.client_id} already exists. Force overwrite with overwrite=True"
            )
        try:
            client_dict = client.dict()
            client_dict.pop("invocation_id", None)

            self._collection.replace_one(
                {"client_id": client.client_id}, client_dict, upsert=True
            )
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


class ClientControlDao(MongoDbDao):
    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "client_controls",
        database: str = DATABASE,
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
        local_controls: SerializedParameters,
        overwrite: bool = True,
    ) -> Any:

        if (
            not overwrite
            and self._collection.find_one(
                {"session_id": session_id, "client_id": client_id}
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Local controls for session {session_id} and round {client_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(
                bson.encode(local_controls.dict()), encoding="utf-8"
            )

            self._collection.replace_one(
                {"session_id": session_id, "client_id": client_id},
                {
                    "session_id": session_id,
                    "client_id": client_id,
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
        client_id: str,
    ) -> SerializedParameters:
        try:
            obj_dict = self._collection.find_one(
                filter={"session_id": session_id, "client_id": client_id},
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Local controls for session {session_id} and client {client_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded Local controls for session {session_id} "
                f"and client {client_id} malformed."
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} and client {client_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(parameter_file.read()))
        finally:
            parameter_file.close()

    @wrap_pymongo_errors
    def delete_local_controls(self, session_id: str):
        try:
            result_dicts = iter(
                self._collection.find(
                    {
                        "session_id": session_id,
                    },
                )
            )
            for result_dict in result_dicts:
                if not result_dict or "file_id" not in result_dict:
                    continue
                self._gridfs.delete(file_id=result_dict["file_id"])
            self._collection.delete_many(
                {
                    "session_id": session_id,
                }
            )
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e


class ParameterDao(MongoDbDao):
    """Store global model parameters in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "parameters",
        database: str = DATABASE,
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
        global_controls: SerializedParameters = None,
        overwrite: bool = True,
    ) -> Any:

        if (
            not overwrite
            and self._collection.find_one(
                {"session_id": session_id, "round_id": round_id}
            )
            is not None
        ):
            raise DocumentAlreadyExistsException(
                f"Parameters for session {session_id} and round {round_id} already exist. "
                f"Force overwrite with overwrite=True"
            )
        try:
            file_id = self._gridfs.put(bson.encode(params.dict()), encoding="utf-8")
            controls_file_id = (
                self._gridfs.put(bson.encode(global_controls.dict()), encoding="utf-8")
                if global_controls
                else None
            )

            self._collection.replace_one(
                {"session_id": session_id, "round_id": round_id},
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "file_id": file_id,
                    "controls_file_id": controls_file_id,
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
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} and round {round_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} "
                f"and round {round_id} malformed."
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
    def load_controls(
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
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} and round {round_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded global controls for session {session_id} "
                f"and round {round_id} malformed."
            )
        controls_file = self._gridfs.find_one({"_id": obj_dict["controls_file_id"]})
        if not controls_file:
            raise DocumentNotLoadedException(
                f"GridFS file with global controls for session {session_id} and round {round_id} not found"
            )
        try:
            return SerializedParameters.parse_obj(bson.decode(controls_file.read()))
        finally:
            controls_file.close()

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
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if obj_dict is None:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if "file_id" not in obj_dict:
            raise PersistenceValueError(
                f"Loaded parameters for session {session_id} " f"Expected key file_id"
            )
        parameter_file = self._gridfs.find_one({"_id": obj_dict["file_id"]})
        if not parameter_file:
            raise DocumentNotLoadedException(
                f"GridFS file with parameters for session {session_id} not found"
            )
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
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found"
            )
        if obj_dict is None or "round_id" not in obj_dict:
            raise DocumentNotLoadedException(
                f"Parameters for session {session_id} not found or malformed"
            )
        return int(obj_dict["round_id"])


class ModelDao(MongoDbDao):
    """Store clients  in a mongodb database"""

    def __init__(
        self,
        db: Union[str, MongodbConnectionConfig, pymongo.MongoClient],
        collection: str = "models",
        database: str = DATABASE,
    ):

        super().__init__(
            db=db,
            collection=collection,
            database=database,
        )

    @wrap_pymongo_errors
    def save(
        self, session_id: str, model: SerializedModel, overwrite: bool = True
    ) -> Any:
        if (
            not overwrite
            and self._collection.find_one({"session_id": session_id}) is not None
        ):
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
            obj_dict = (
                obj_dict["model"]
                if obj_dict is not None and "model" in obj_dict
                else None
            )

            if obj_dict is None:
                raise DocumentNotLoadedException(f"Client with id {id} not found")

            return SerializedModel.parse_obj(obj_dict)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        except KeyError:
            raise PersistenceValueError(
                f"Loaded model architecture for session {session_id} malformed. Expected key parameters"
            )
