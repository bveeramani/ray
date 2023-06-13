import enum
import logging
import inspect
import os
import asyncio
from functools import wraps
from typing import Dict, List, Optional

import grpc

from ray._private import ray_constants
from ray.core.generated import (
    gcs_service_pb2,
    gcs_service_pb2_grpc,
)

from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
    ActorTableData,
    AvailableResources,
    ErrorTableData,
    GcsEntry,
    GcsNodeInfo,
    JobTableData,
    ObjectTableData,
    PlacementGroupTableData,
    PubSubMessage,
    ResourceDemand,
    ResourceLoad,
    ResourceMap,
    ResourcesData,
    ResourceTableData,
    ResourceUsageBatchData,
    TablePrefix,
    TablePubsub,
    TaskEvents,
    WorkerTableData,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ActorTableData",
    "GcsNodeInfo",
    "AvailableResources",
    "JobTableData",
    "JobConfig",
    "ErrorTableData",
    "ErrorType",
    "GcsEntry",
    "ResourceUsageBatchData",
    "ResourcesData",
    "ObjectTableData",
    "TablePrefix",
    "TablePubsub",
    "TaskEvents",
    "ResourceDemand",
    "ResourceLoad",
    "ResourceMap",
    "ResourceTableData",
    "PubSubMessage",
    "WorkerTableData",
    "PlacementGroupTableData",
]


WORKER = 0
DRIVER = 1

# Cap messages at 512MB
_MAX_MESSAGE_LENGTH = 512 * 1024 * 1024
# Send keepalive every 60s
_GRPC_KEEPALIVE_TIME_MS = 60 * 1000
# Keepalive should be replied < 60s
_GRPC_KEEPALIVE_TIMEOUT_MS = 60 * 1000

# Also relying on these defaults:
# grpc.keepalive_permit_without_calls=0: No keepalive without inflight calls.
# grpc.use_local_subchannel_pool=0: Subchannels are shared.
_GRPC_OPTIONS = [
    *ray_constants.GLOBAL_GRPC_OPTIONS,
    ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
    ("grpc.keepalive_time_ms", _GRPC_KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", _GRPC_KEEPALIVE_TIMEOUT_MS),
]


def create_gcs_channel(address: str, aio=False):
    """Returns a GRPC channel to GCS.

    Args:
        address: GCS address string, e.g. ip:port
        aio: Whether using grpc.aio
    Returns:
        grpc.Channel or grpc.aio.Channel to GCS
    """
    from ray._private.utils import init_grpc_channel

    return init_grpc_channel(address, options=_GRPC_OPTIONS, asynchronous=aio)


# This global variable is used for testing only
_called_freq = {}


def _auto_reconnect(f):
    # This is for testing to count the frequence
    # of gcs call
    if inspect.iscoroutinefunction(f):

        @wraps(f)
        async def wrapper(self, *args, **kwargs):
            if "TEST_RAY_COLLECT_KV_FREQUENCY" in os.environ:
                global _called_freq
                name = f.__name__
                if name not in _called_freq:
                    _called_freq[name] = 0
                _called_freq[name] += 1

            remaining_retry = self._nums_reconnect_retry
            while True:
                try:
                    return await f(self, *args, **kwargs)
                except grpc.RpcError as e:
                    if e.code() in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.UNKNOWN,
                    ):
                        if remaining_retry <= 0:
                            logger.error(
                                "Failed to connect to GCS. Please check"
                                " `gcs_server.out` for more details."
                            )
                            raise
                        logger.debug(
                            "Failed to send request to gcs, reconnecting. " f"Error {e}"
                        )
                        try:
                            self._connect()
                        except Exception:
                            logger.error(f"Connecting to gcs failed. Error {e}")
                        await asyncio.sleep(1)
                        remaining_retry -= 1
                        continue
                    raise

        return wrapper
    else:

        raise NotImplementedError(
            "This code moved to Cython, see "
            "https://github.com/ray-project/ray/pull/33769"
        )


class GcsChannel:
    def __init__(self, gcs_address: Optional[str] = None, aio: bool = False):
        self._gcs_address = gcs_address
        self._aio = aio

    @property
    def address(self):
        return self._gcs_address

    def connect(self):
        # GCS server uses a cached port, so it should use the same port after
        # restarting. This means GCS address should stay the same for the
        # lifetime of the Ray cluster.
        self._channel = create_gcs_channel(self._gcs_address, self._aio)

    def channel(self):
        return self._channel


class GcsCode(enum.IntEnum):
    # corresponding to ray/src/ray/common/status.h
    OK = 0
    NotFound = 17
    GrpcUnavailable = 26


class GcsAioClient:
    def __init__(
        self,
        channel: Optional[GcsChannel] = None,
        address: Optional[str] = None,
        nums_reconnect_retry: int = 5,
    ):
        if channel is None:
            assert isinstance(address, str)
            channel = GcsChannel(gcs_address=address, aio=True)
        assert isinstance(channel, GcsChannel)
        assert channel._aio is True
        self._channel = channel
        self._connect()
        self._nums_reconnect_retry = nums_reconnect_retry

    @property
    def channel(self):
        return self._channel

    def _connect(self):
        self._channel.connect()
        self._kv_stub = gcs_service_pb2_grpc.InternalKVGcsServiceStub(
            self._channel.channel()
        )
        self._node_info_stub = gcs_service_pb2_grpc.NodeInfoGcsServiceStub(
            self._channel.channel()
        )
        self._job_info_stub = gcs_service_pb2_grpc.JobInfoGcsServiceStub(
            self._channel.channel()
        )
        self._actor_info_stub = gcs_service_pb2_grpc.ActorInfoGcsServiceStub(
            self._channel.channel()
        )

    @_auto_reconnect
    async def check_alive(
        self, node_ips: List[bytes], timeout: Optional[float] = None
    ) -> List[bool]:
        req = gcs_service_pb2.CheckAliveRequest(raylet_address=node_ips)
        reply = await self._node_info_stub.CheckAlive(req, timeout=timeout)

        if reply.status.code != GcsCode.OK:
            raise RuntimeError(
                f"GCS running at {self._channel.address} is unhealthy: {reply.status}"
            )
        return list(reply.raylet_alive)

    @_auto_reconnect
    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[bytes]:
        logger.debug(f"internal_kv_get {key!r} {namespace!r}")
        req = gcs_service_pb2.InternalKVGetRequest(namespace=namespace, key=key)
        reply = await self._kv_stub.InternalKVGet(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return reply.value
        elif reply.status.code == GcsCode.NotFound:
            return None
        else:
            raise RuntimeError(
                f"Failed to get value for key {key!r} "
                f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def internal_kv_multi_get(
        self,
        keys: List[bytes],
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> Dict[bytes, bytes]:
        logger.debug(f"internal_kv_multi_get {keys!r} {namespace!r}")
        req = gcs_service_pb2.InternalKVMultiGetRequest(namespace=namespace, keys=keys)
        reply = await self._kv_stub.InternalKVMultiGet(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return {entry.key: entry.value for entry in reply.results}
        else:
            raise RuntimeError(
                f"Failed to get value for keys {keys!r} "
                f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def internal_kv_put(
        self,
        key: bytes,
        value: bytes,
        overwrite: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        """Put a key-value pair into the GCS.

        Args:
            key: The key to put.
            value: The value to put.
            overwrite: Whether to overwrite the value if the key already exists.
            namespace: The namespace to put the key-value pair into.
            timeout: The timeout in seconds.

        Returns:
            The number of keys added. If overwrite is True, this will be 1 if the
                key was added and 0 if the key was updated. If overwrite is False,
                this will be 1 if the key was added and 0 if the key already exists.
        """
        logger.debug(f"internal_kv_put {key!r} {value!r} {overwrite} {namespace!r}")
        req = gcs_service_pb2.InternalKVPutRequest(
            namespace=namespace,
            key=key,
            value=value,
            overwrite=overwrite,
        )
        reply = await self._kv_stub.InternalKVPut(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return reply.added_num
        else:
            raise RuntimeError(
                f"Failed to put value {value!r} to key {key!r} "
                f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def internal_kv_del(
        self,
        key: bytes,
        del_by_prefix: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(f"internal_kv_del {key!r} {del_by_prefix} {namespace!r}")
        req = gcs_service_pb2.InternalKVDelRequest(
            namespace=namespace, key=key, del_by_prefix=del_by_prefix
        )
        reply = await self._kv_stub.InternalKVDel(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return reply.deleted_num
        else:
            raise RuntimeError(
                f"Failed to delete key {key!r} " f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        logger.debug(f"internal_kv_exists {key!r} {namespace!r}")
        req = gcs_service_pb2.InternalKVExistsRequest(namespace=namespace, key=key)
        reply = await self._kv_stub.InternalKVExists(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return reply.exists
        else:
            raise RuntimeError(
                f"Failed to check existence of key {key!r} "
                f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes]:
        logger.debug(f"internal_kv_keys {prefix!r} {namespace!r}")
        req = gcs_service_pb2.InternalKVKeysRequest(namespace=namespace, prefix=prefix)
        reply = await self._kv_stub.InternalKVKeys(req, timeout=timeout)
        if reply.status.code == GcsCode.OK:
            return reply.results
        else:
            raise RuntimeError(
                f"Failed to list prefix {prefix!r} "
                f"due to error {reply.status.message}"
            )

    @_auto_reconnect
    async def get_all_job_info(
        self, timeout: Optional[float] = None
    ) -> gcs_service_pb2.GetAllJobInfoReply:
        req = gcs_service_pb2.GetAllJobInfoRequest()
        reply = await self._job_info_stub.GetAllJobInfo(req, timeout=timeout)
        return reply

    @_auto_reconnect
    async def get_named_actor_info(
        self,
        actor_name: str,
        ray_namespace: str = "",
        timeout: Optional[float] = None,
    ) -> gcs_service_pb2.GetNamedActorInfoReply:
        req = gcs_service_pb2.GetNamedActorInfoRequest(
            name=actor_name, ray_namespace=ray_namespace
        )
        reply = await self._actor_info_stub.GetNamedActorInfo(req, timeout=timeout)
        return reply


def cleanup_redis_storage(
    host: str, port: int, password: str, use_ssl: bool, storage_namespace: str
):
    """This function is used to cleanup the storage. Before we having
    a good design for storage backend, it can be used to delete the old
    data. It support redis cluster and non cluster mode.

    Args:
       host: The host address of the Redis.
       port: The port of the Redis.
       password: The password of the Redis.
       use_ssl: Whether to encrypt the connection.
       storage_namespace: The namespace of the storage to be deleted.
    """

    from ray._raylet import del_key_from_storage  # type: ignore

    if not isinstance(host, str):
        raise ValueError("Host must be a string")

    if not isinstance(password, str):
        raise ValueError("Password must be a string")

    if port < 0:
        raise ValueError(f"Invalid port: {port}")

    if not isinstance(use_ssl, bool):
        raise TypeError("use_ssl must be a boolean")

    if not isinstance(storage_namespace, str):
        raise ValueError("storage namespace must be a string")

    # Right now, GCS store all data into a hash set key by storage_namespace.
    # So we only need to delete the specific key to cleanup the cluster.
    return del_key_from_storage(host, port, password, use_ssl, storage_namespace)
