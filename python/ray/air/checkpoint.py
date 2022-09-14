import contextlib
from dataclasses import dataclass
import io
import logging
import os
import shutil
import tarfile
import tempfile
import traceback
from pathlib import Path
import platform
from typing import Any, Dict, Iterator, Optional, Tuple, Type, Union, TYPE_CHECKING
import uuid

import ray
from ray import cloudpickle as pickle
from ray.air._internal.checkpointing import load_preprocessor_from_dir
from ray.air._internal.filelock import TempFileLock
from ray.air._internal.remote_storage import (
    download_from_uri,
    fs_hint,
    is_non_local_path_uri,
    read_file_from_uri,
    upload_to_uri,
)
from ray.air.constants import PREPROCESSOR_KEY
from ray.util.annotations import DeveloperAPI, PublicAPI


if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor


_DICT_CHECKPOINT_FILE_NAME = "dict_checkpoint.pkl"
_CHECKPOINT_METADATA_FILE_NAME = ".metadata.pkl"
_DICT_CHECKPOINT_ADDITIONAL_FILE_KEY = "_ray_additional_checkpoint_files"
_METADATA_CHECKPOINT_SUFFIX = ".meta.pkl"
_FS_CHECKPOINT_KEY = "fs_checkpoint"
_BYTES_DATA_KEY = "bytes_data"
_CHECKPOINT_DIR_PREFIX = "checkpoint_tmp_"

logger = logging.getLogger(__name__)


@DeveloperAPI
@dataclass
class CheckpointMetadata:
    checkpoint_type: Type["Checkpoint"]
    checkpoint_state: Dict[str, Any]
    ray_version: str = ray.__version__


@DeveloperAPI
class CheckpointDict(dict):
    def __init__(self, *args, metadata: CheckpointMetadata, **kwargs):
        self.metadata = metadata
        super().__init__(*args, **kwargs)


@PublicAPI(stability="beta")
class Checkpoint:
    """Ray AIR Checkpoint.

    An AIR Checkpoint are a common interface for accessing models across
    different AIR components and libraries. A Checkpoint can have its data
    represented in one of three ways:

    - as a directory on local (on-disk) storage
    - as a directory on an external storage (e.g., cloud storage)
    - as an in-memory dictionary

    The Checkpoint object also has methods to translate between different checkpoint
    storage locations. These storage representations provide flexibility in
    distributed environments, where you may want to recreate an instance of
    the same model on multiple nodes or across different Ray clusters.

    Example:

    .. code-block:: python

        from ray.air.checkpoint import Checkpoint

        # Create checkpoint data dict
        checkpoint_data = {"data": 123}

        # Create checkpoint object from data
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        # Save checkpoint to a directory on the file system.
        path = checkpoint.to_directory()

        # This path can then be passed around,
        # # e.g. to a different function or a different script.
        # You can also use `checkpoint.to_uri/from_uri` to
        # read from/write to cloud storage

        # In another function or script, recover Checkpoint object from path
        checkpoint = Checkpoint.from_directory(path)

        # Convert into dictionary again
        recovered_data = checkpoint.to_dict()

        # It is guaranteed that the original data has been recovered
        assert recovered_data == checkpoint_data

    Checkpoints can be used to instantiate a :class:`Predictor`,
    :class:`BatchPredictor`, or :class:`PredictorDeployment` class.

    The constructor is a private API, instead the ``from_`` methods should
    be used to create checkpoint objects
    (e.g. ``Checkpoint.from_directory()``).

    **Other implementation notes:**

    When converting between different checkpoint formats, it is guaranteed
    that a full round trip of conversions (e.g. directory --> dict -->
    directory) will recover the original checkpoint data. There are no guarantees made
    about compatibility of intermediate representations.

    New data can be added to a Checkpoint
    during conversion. Consider the following conversion:
    directory --> dict (adding dict["foo"] = "bar")
    --> directory --> dict (expect to see dict["foo"] = "bar"). Note that
    the second directory will contain pickle files with the serialized additional
    field data in them.

    Similarly with a dict as a source: dict --> directory (add file "foo.txt")
    --> dict --> directory (will have "foo.txt" in it again). Note that the second
    dict representation will contain an extra field with the serialized additional
    files in it.

    Checkpoints can be pickled and sent to remote processes.
    Please note that checkpoints pointing to local directories will be
    pickled as data representations, so the full checkpoint data will be
    contained in the checkpoint object. If you want to avoid this,
    consider passing only the checkpoint directory to the remote task
    and re-construct your checkpoint object in that function. Note that
    this will only work if the "remote" task is scheduled on the
    same node or a node that also has access to the local data path (e.g.
    on a shared file system like NFS).
    """

    _SERIALIZED_ATTRS = ()

    @DeveloperAPI
    def __init__(
        self,
        local_path: Optional[str] = None,
        data_dict: Optional[dict] = None,
        uri: Optional[str] = None,
    ):
        # First, resolve file:// URIs to local paths
        if uri:
            local_path = _get_local_path(uri)
            if local_path:
                uri = None

        # Only one data type can be set at any time
        if local_path:
            assert not data_dict and not uri
            if not isinstance(local_path, (str, os.PathLike)) or not os.path.exists(
                local_path
            ):
                raise RuntimeError(
                    f"Cannot create checkpoint from path as it does "
                    f"not exist on local node: {local_path}"
                )
            elif not os.path.isdir(local_path):
                raise RuntimeError(
                    f"Cannot create checkpoint from path as it does "
                    f"not point to a directory: {local_path}. If your checkpoint "
                    f"is a single file, consider passing the enclosing directory "
                    f"instead."
                )
        elif data_dict:
            assert not local_path and not uri
            if not isinstance(data_dict, dict):
                raise RuntimeError(
                    f"Cannot create checkpoint from dict as no "
                    f"dict was passed: {data_dict}"
                )
        elif uri:
            assert not local_path and not data_dict
            resolved = _get_external_path(uri)
            if not resolved:
                raise RuntimeError(
                    f"Cannot create checkpoint from URI as it is not "
                    f"supported: {resolved}"
                )
            uri = resolved
        else:
            raise ValueError("Cannot create checkpoint without data.")

        self._local_path: Optional[str] = local_path
        self._data_dict: Optional[Dict[str, Any]] = data_dict
        self._uri: Optional[str] = uri

        self._uuid = uuid.uuid4()

    def __repr__(self):
        parameter, argument = self.get_internal_representation()
        return f"{self.__class__.__name__}({parameter}={argument})"

    @DeveloperAPI
    @property
    def metadata(self) -> CheckpointMetadata:
        return CheckpointMetadata(
            checkpoint_type=self.__class__,
            checkpoint_state={
                attr: getattr(self, attr) for attr in self._SERIALIZED_ATTRS
            },
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "Checkpoint":
        """Create a checkpoint from the given byte string.

        Args:
            data: Data object containing pickled checkpoint data.

        Returns:
            Checkpoint: checkpoint object.
        """
        bytes_data = pickle.loads(data)
        if isinstance(bytes_data, dict):
            data_dict = bytes_data
        else:
            data_dict = {_BYTES_DATA_KEY: bytes_data}
        return cls.from_dict(data_dict)

    def to_bytes(self) -> bytes:
        """Return Checkpoint serialized as bytes object.

        Returns:
            bytes: Bytes object containing checkpoint data.
        """
        # Todo: Add support for stream in the future (to_bytes(file_like))
        data_dict = self.to_dict()
        if "bytes_data" in data_dict:
            return data_dict["bytes_data"]
        return pickle.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """Create checkpoint object from dictionary.

        Args:
            data: Dictionary containing checkpoint data.

        Returns:
            Checkpoint: checkpoint object.
        """
        state = {}
        if isinstance(data, CheckpointDict):
            cls._check_type_compatability(data.metadata.checkpoint_type)
            cls = data.metadata.checkpoint_type
            state = data.metadata.checkpoint_state

        checkpoint = cls(data_dict=data)
        checkpoint.__dict__.update(state)

        return checkpoint

    def to_dict(self) -> dict:
        """Return checkpoint data as dictionary.

        .. note::
            :meth:`~Checkpoint.to_dict` returns a ``dict`` subclass that contains
            information about the checkpoint type. This ``dict`` subclass is
            functionally identical to the built-in ``dict``.

        Returns:
            dict: Dictionary containing checkpoint data.
        """
        if self._data_dict:
            # If the checkpoint data is already a dict, return
            checkpoint_data = self._data_dict
        elif self._local_path or self._uri:
            # Else, checkpoint is either on FS or external storage
            with self.as_directory() as local_path:
                checkpoint_data_path = os.path.join(
                    local_path, _DICT_CHECKPOINT_FILE_NAME
                )
                if os.path.exists(checkpoint_data_path):
                    # If we are restoring a dict checkpoint, load the dict
                    # from the checkpoint file.
                    with open(checkpoint_data_path, "rb") as f:
                        checkpoint_data = pickle.load(f)

                    # If there are additional files in the directory, add them as
                    # _DICT_CHECKPOINT_ADDITIONAL_FILE_KEY
                    additional_files = {}
                    for file_or_dir in os.listdir(local_path):
                        if file_or_dir in [
                            ".",
                            "..",
                            _DICT_CHECKPOINT_FILE_NAME,
                            _CHECKPOINT_METADATA_FILE_NAME,
                        ]:
                            continue

                        additional_files[file_or_dir] = _pack(
                            os.path.join(local_path, file_or_dir)
                        )

                    if additional_files:
                        checkpoint_data[
                            _DICT_CHECKPOINT_ADDITIONAL_FILE_KEY
                        ] = additional_files

                else:
                    files = [
                        f
                        for f in os.listdir(local_path)
                        if os.path.isfile(os.path.join(local_path, f))
                        and f.endswith(_METADATA_CHECKPOINT_SUFFIX)
                    ]
                    metadata = {}
                    for file in files:
                        with open(os.path.join(local_path, file), "rb") as f:
                            key = file[: -len(_METADATA_CHECKPOINT_SUFFIX)]
                            value = pickle.load(f)
                            metadata[key] = value

                    data = _pack(local_path)

                    checkpoint_data = {
                        _FS_CHECKPOINT_KEY: data,
                    }
                    checkpoint_data.update(metadata)
        else:
            raise RuntimeError(f"Empty data for checkpoint {self}")

        return CheckpointDict(checkpoint_data, metadata=self.metadata)

    @classmethod
    def from_directory(cls, path: str) -> "Checkpoint":
        """Create checkpoint object from directory.

        Args:
            path: Directory containing checkpoint data. The caller promises to
                not delete the directory (gifts ownership of the directory to this
                Checkpoint).

        Returns:
            Checkpoint: checkpoint object.
        """
        state = {}

        checkpoint_metadata_path = os.path.join(path, _CHECKPOINT_METADATA_FILE_NAME)
        if os.path.exists(checkpoint_metadata_path):
            with open(checkpoint_metadata_path, "rb") as file:
                metadata = pickle.load(file)
                cls._check_type_compatability(metadata.checkpoint_type)
                cls = metadata.checkpoint_type
                state = metadata.checkpoint_state

        checkpoint = cls(local_path=path)
        checkpoint.__dict__.update(state)

        return checkpoint

    @classmethod
    def from_checkpoint(cls, other: "Checkpoint") -> "Checkpoint":
        """Create a checkpoint from a generic :py:class:`Checkpoint`.

        This method can be used to create a framework-specific checkpoint from a
        generic :py:class:`Checkpoint` object.

        Examples:
            >>> result = TorchTrainer.fit(...)  # doctest: +SKIP
            >>> checkpoint = TorchCheckpoint.from_checkpoint(result.checkpoint)  # doctest: +SKIP # noqa: E501
            >>> model = checkpoint.get_model()  # doctest: +SKIP
            Linear(in_features=1, out_features=1, bias=True)
        """
        return cls(
            local_path=other._local_path,
            data_dict=other._data_dict,
            uri=other._uri,
        )

    @classmethod
    def _check_type_compatability(cls, serialized_type: Type["Checkpoint"]):
        if not isinstance(serialized_type, cls):
            raise ValueError(
                "The checkpoint data you passed in was created by a "
                f"`{serialized_type.__name__}` object, but "
                f"`{serialized_type.__name__}` isn't compatible with {cls.__name__}`."
            )

    def _get_temporary_checkpoint_dir(self) -> str:
        """Return the name for the temporary checkpoint dir."""
        tmp_dir_path = tempfile.gettempdir()
        checkpoint_dir_name = _CHECKPOINT_DIR_PREFIX + self._uuid.hex
        if platform.system() == "Windows":
            # Max path on Windows is 260 chars, -1 for joining \
            # Also leave a little for the del lock
            del_lock_name = _get_del_lock_path("")
            checkpoint_dir_name = (
                _CHECKPOINT_DIR_PREFIX
                + self._uuid.hex[
                    -259
                    + len(_CHECKPOINT_DIR_PREFIX)
                    + len(tmp_dir_path)
                    + len(del_lock_name) :
                ]
            )
            if not checkpoint_dir_name.startswith(_CHECKPOINT_DIR_PREFIX):
                raise RuntimeError(
                    "Couldn't create checkpoint directory due to length "
                    "constraints. Try specifing a shorter checkpoint path."
                )
        return os.path.join(tmp_dir_path, checkpoint_dir_name)

    def _to_directory(self, path: str) -> None:
        if self._data_dict:
            data_dict = self.to_dict()
            if _FS_CHECKPOINT_KEY in data_dict:
                for key in data_dict.keys():
                    if key == _FS_CHECKPOINT_KEY:
                        continue
                    metadata_path = os.path.join(
                        path, f"{key}{_METADATA_CHECKPOINT_SUFFIX}"
                    )
                    with open(metadata_path, "wb") as f:
                        pickle.dump(data_dict[key], f)
                # This used to be a true fs checkpoint, so restore
                _unpack(data_dict[_FS_CHECKPOINT_KEY], path)
            else:
                # This is a dict checkpoint.
                # First, restore any additional files
                additional_files = data_dict.pop(
                    _DICT_CHECKPOINT_ADDITIONAL_FILE_KEY, {}
                )
                for file, content in additional_files.items():
                    _unpack(stream=content, path=os.path.join(path, file))

                # Then dump data into checkpoint.pkl
                checkpoint_data_path = os.path.join(path, _DICT_CHECKPOINT_FILE_NAME)
                with open(checkpoint_data_path, "wb") as f:
                    pickle.dump(data_dict, f)
        else:
            # This is either a local fs, remote node fs, or external fs
            local_path = self._local_path
            external_path = _get_external_path(self._uri)
            if local_path:
                if local_path != path:
                    # If this exists on the local path, just copy over
                    if path and os.path.exists(path):
                        shutil.rmtree(path)
                    shutil.copytree(local_path, path)
            elif external_path:
                # If this exists on external storage (e.g. cloud), download
                download_from_uri(uri=external_path, local_path=path, filelock=False)
            else:
                raise RuntimeError(
                    f"No valid location found for checkpoint {self}: {self._uri}"
                )

        checkpoint_metadata_path = os.path.join(path, _CHECKPOINT_METADATA_FILE_NAME)
        with open(checkpoint_metadata_path, "wb") as file:
            pickle.dump(self.metadata, file)

    def to_directory(self, path: Optional[str] = None) -> str:
        """Write checkpoint data to directory.

        Args:
            path: Target directory to restore data in. If not specified,
                will create a temporary directory.

        Returns:
            str: Directory containing checkpoint data.
        """
        user_provided_path = path is not None
        path = path if user_provided_path else self._get_temporary_checkpoint_dir()
        path = os.path.normpath(path)

        _make_dir(path, acquire_del_lock=not user_provided_path)

        try:
            # Timeout 0 means there will be only one attempt to acquire
            # the file lock. If it cannot be aquired, a TimeoutError
            # will be thrown.
            with TempFileLock(f"{path}.lock", timeout=0):
                self._to_directory(path)
        except TimeoutError:
            # if the directory is already locked, then wait but do not do anything.
            with TempFileLock(f"{path}.lock", timeout=-1):
                pass
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Checkpoint directory {path} does not exist, "
                    "even though it should have been created by "
                    "another process. Please raise an issue on GitHub: "
                    "https://github.com/ray-project/ray/issues"
                )

        return path

    @contextlib.contextmanager
    def as_directory(self) -> Iterator[str]:
        """Return checkpoint directory path in a context.

        This function makes checkpoint data available as a directory while avoiding
        unnecessary copies and left-over temporary data.

        If the checkpoint is already a directory checkpoint, it will return
        the existing path. If it is not, it will create a temporary directory,
        which will be deleted after the context is exited.

        Users should treat the returned checkpoint directory as read-only and avoid
        changing any data within it, as it might get deleted when exiting the context.

        Example:

        .. code-block:: python

            with checkpoint.as_directory() as checkpoint_dir:
                # Do some read-only processing of files within checkpoint_dir
                pass

            # At this point, if a temporary directory was created, it will have
            # been deleted.

        """
        if self._local_path:
            yield self._local_path
        else:
            temp_dir = self.to_directory()
            del_lock_path = _get_del_lock_path(temp_dir)
            yield temp_dir

            # Cleanup
            try:
                os.remove(del_lock_path)
            except Exception:
                logger.warning(
                    f"Could not remove {del_lock_path} deletion file lock. "
                    f"Traceback:\n{traceback.format_exc()}"
                )

            # In the edge case (process crash before del lock file is removed),
            # we do not remove the directory at all.
            # Since it's in /tmp, this is not that big of a deal.
            # check if any lock files are remaining
            temp_dir_base_name = Path(temp_dir).name
            if not list(
                Path(temp_dir).parent.glob(_get_del_lock_path(temp_dir_base_name, "*"))
            ):
                try:
                    # Timeout 0 means there will be only one attempt to acquire
                    # the file lock. If it cannot be aquired, a TimeoutError
                    # will be thrown.
                    with TempFileLock(f"{temp_dir}.lock", timeout=0):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except TimeoutError:
                    pass

    @classmethod
    def from_uri(cls, uri: str) -> "Checkpoint":
        """Create checkpoint object from location URI (e.g. cloud storage).

        Valid locations currently include AWS S3 (``s3://``),
        Google cloud storage (``gs://``), HDFS (``hdfs://``), and
        local files (``file://``).

        Args:
            uri: Source location URI to read data from.

        Returns:
            Checkpoint: checkpoint object.
        """
        state = {}
        try:
            checkpoint_metadata_uri = os.path.join(uri, _CHECKPOINT_METADATA_FILE_NAME)
            metadata = pickle.loads(read_file_from_uri(checkpoint_metadata_uri))
            cls._check_type_compatability(metadata.checkpoint_type)
            cls = metadata.checkpoint_type
            state = metadata.checkpoint_state
        except FileNotFoundError:
            pass

        checkpoint = cls(uri=uri)
        checkpoint.__dict__.update(state)

        return checkpoint

    def to_uri(self, uri: str) -> str:
        """Write checkpoint data to location URI (e.g. cloud storage).

        Args:
            uri: Target location URI to write data to.

        Returns:
            str: Cloud location containing checkpoint data.
        """
        if uri.startswith("file://"):
            local_path = uri[7:]
            return self.to_directory(local_path)

        if not is_non_local_path_uri(uri):
            raise RuntimeError(
                f"Cannot upload checkpoint to URI: Provided URI "
                f"does not belong to a registered storage provider: `{uri}`. "
                f"Hint: {fs_hint(uri)}"
            )

        with self.as_directory() as local_path:
            checkpoint_metadata_path = os.path.join(
                local_path, _CHECKPOINT_METADATA_FILE_NAME
            )
            with open(checkpoint_metadata_path, "wb") as file:
                pickle.dump(self.metadata, file)

            upload_to_uri(local_path=local_path, uri=uri)

        return uri

    @DeveloperAPI
    def get_internal_representation(
        self,
    ) -> Tuple[str, Union[dict, str]]:
        """Return tuple of (type, data) for the internal representation.

        The internal representation can be used e.g. to compare checkpoint
        objects for equality or to access the underlying data storage.

        The returned type is a string and one of
        ``["local_path", "data_dict", "uri"]``.

        The data is the respective data value.

        Note that paths converted from ``file://...`` will be returned
        as ``local_path`` (without the ``file://`` prefix) and not as ``uri``.

        Returns:
            Tuple of type and data.
        """
        if self._local_path:
            return "local_path", self._local_path
        elif self._data_dict:
            return "data_dict", self._data_dict
        elif self._uri:
            return "uri", self._uri
        else:
            raise RuntimeError(
                "Cannot get internal representation of empty checkpoint."
            )

    def __getstate__(self):
        if self._local_path:
            blob = self.to_bytes()
            return self.__class__.from_bytes(blob).__getstate__()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __fspath__(self):
        raise TypeError(
            "You cannot use `air.Checkpoint` objects directly as paths. "
            "Use `Checkpoint.to_directory()` or `Checkpoint.as_directory()` instead."
        )

    def get_preprocessor(self) -> Optional["Preprocessor"]:
        """Return the saved preprocessor, if one exists."""

        # The preprocessor will either be stored in an in-memory dict or
        # written to storage. In either case, it will use the PREPROCESSOR_KEY key.

        # First try converting to dictionary.
        checkpoint_dict = self.to_dict()
        preprocessor = checkpoint_dict.get(PREPROCESSOR_KEY, None)

        if preprocessor is None:
            # Fallback to reading from directory.
            with self.as_directory() as checkpoint_path:
                preprocessor = load_preprocessor_from_dir(checkpoint_path)

        return preprocessor


def _get_local_path(path: Optional[str]) -> Optional[str]:
    """Check if path is a local path. Otherwise return None."""
    if path is None or is_non_local_path_uri(path):
        return None
    if path.startswith("file://"):
        path = path[7:]
    if os.path.exists(path):
        return path
    return None


def _get_external_path(path: Optional[str]) -> Optional[str]:
    """Check if path is an external path. Otherwise return None."""
    if not isinstance(path, str) or not is_non_local_path_uri(path):
        return None
    return path


def _pack(path: str) -> bytes:
    """Pack directory in ``path`` into an archive, return as bytes string."""
    stream = io.BytesIO()

    def filter_function(tarinfo):
        if tarinfo.name.endswith(_METADATA_CHECKPOINT_SUFFIX):
            return None
        else:
            return tarinfo

    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
        tar.add(path, arcname="", filter=filter_function)

    return stream.getvalue()


def _unpack(stream: bytes, path: str) -> str:
    """Unpack archive in bytes string into directory in ``path``."""
    with tarfile.open(fileobj=io.BytesIO(stream)) as tar:
        tar.extractall(path)
    return path


def _get_del_lock_path(path: str, pid: str = None) -> str:
    """Get the path to the deletion lock file."""
    pid = pid if pid is not None else os.getpid()
    return f"{path}.del_lock_{pid}"


def _make_dir(path: str, acquire_del_lock: bool = True) -> None:
    """Create the temporary checkpoint dir in ``path``."""
    if acquire_del_lock:
        # Each process drops a deletion lock file it then cleans up.
        # If there are no lock files left, the last process
        # will remove the entire directory.
        del_lock_path = _get_del_lock_path(path)
        open(del_lock_path, "a").close()

    os.makedirs(path, exist_ok=True)
