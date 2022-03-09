import tempfile
from dataclasses import dataclass
import io
import logging
import os
import random

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from ray import train
from ray.train.accelerator import Accelerator
from ray.train.backend import BackendConfig, Backend, EncodedData
from ray.train.constants import PYTORCH_PROFILER_KEY
from ray.train.session import get_accelerator, set_accelerator
from ray.train.worker_group import WorkerGroup
from ray.train.utils import get_address_and_port
from ray.util import PublicAPI

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DistributedSampler,
    DataLoader,
    IterableDataset,
    SequentialSampler,
)

try:
    from torch.profiler import profile
except ImportError:
    profile = None

logger = logging.getLogger(__name__)


class TorchAccelerator(Accelerator):
    """A utility that implements methods to accelerate PyTorch training."""

    def __init__(self):
        self._seed = None

    def prepare_model(
        self,
        model: torch.nn.Module,
        move_to_device: bool = True,
        wrap_ddp: bool = True,
        ddp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        """Prepares the model for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            model (torch.nn.Module): A torch model to prepare.
            move_to_device (bool): Whether to move the model to the correct
                device. If set to False, the model needs to manually be moved
                to the correct device.
            wrap_ddp (bool): Whether to wrap models in
                ``DistributedDataParallel``.
            ddp_kwargs (Dict[str, Any]): Args to pass into
                ``DistributedDataParallel`` initialization if ``wrap_ddp`` is
                set to True.
        """
        ddp_kwargs = ddp_kwargs or {}

        rank = train.local_rank()

        device = self.get_device()

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        if move_to_device:
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)
        if wrap_ddp and train.world_size() > 1:
            logger.info("Wrapping provided model in DDP.")
            if torch.cuda.is_available():
                model = DistributedDataParallel(
                    model, device_ids=[rank], output_device=rank, **ddp_kwargs
                )
            else:
                model = DistributedDataParallel(model, **ddp_kwargs)

        return model

    def prepare_data_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        add_dist_sampler: bool = True,
        move_to_device: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Prepares DataLoader for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to
                prepare.
            add_dist_sampler (bool): Whether to add a DistributedSampler to
                the provided DataLoader.
            move_to_device (bool): If set, automatically move the data
                returned by the data loader to the correct device.
        """

        # Only add Distributed Sampler if the following conditions hold:
        # 1. More than one training worker is being used.
        # 2. A DistributedSampler has not already been added by the user.
        # 3. The dataset is not an IterableDataset. Samplers do not worker with
        # IterableDatasets.
        if (
            train.world_size() > 1
            and not isinstance(data_loader.sampler, DistributedSampler)
            and not (
                hasattr(data_loader, "dataset")
                and isinstance(data_loader.dataset, IterableDataset)
            )
            and add_dist_sampler
        ):

            def with_sampler(loader):
                # Automatically set the DistributedSampler

                # If using a sampler, the shuffle attribute in the
                # DataLoader must be set to False.
                # Instead the shuffling is determined by the shuffle attribute
                # in the DistributedSampler.
                # We identify if shuffling is enabled in the passed in
                # DataLoader by seeing if the sampler for the DataLoader is a
                # SequentialSampler.
                shuffle = not isinstance(loader.sampler, SequentialSampler)

                def seed_worker(worker_id):
                    worker_seed = torch.initial_seed() % 2 ** 32
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)

                def seeded_worker_init_fn(worker_init_fn):
                    def wrapper(worker_id):
                        seed_worker(worker_id)
                        worker_init_fn(worker_id)

                    return wrapper

                worker_init_fn = loader.worker_init_fn
                generator = loader.generator
                if self._seed is not None:
                    worker_init_fn = seeded_worker_init_fn(loader.worker_init_fn)
                    generator = torch.Generator()
                    generator.manual_seed(self._seed)

                data_loader_args = {
                    "dataset": loader.dataset,
                    "batch_size": loader.batch_size,
                    "shuffle": False,
                    "num_workers": loader.num_workers,
                    "collate_fn": loader.collate_fn,
                    "pin_memory": loader.pin_memory,
                    "drop_last": loader.drop_last,
                    "timeout": loader.timeout,
                    "worker_init_fn": worker_init_fn,
                    "generator": generator,
                    "sampler": DistributedSampler(loader.dataset, shuffle=shuffle),
                }
                return DataLoader(**data_loader_args)

            data_loader = with_sampler(data_loader)

        if move_to_device:
            device = self.get_device()
            data_loader = _WrappedDataLoader(data_loader, device)

        return data_loader

    def get_device(self) -> torch.device:
        """Gets the correct torch device to use for training."""
        if torch.cuda.is_available():
            rank = train.local_rank()
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        return device

    def enable_reproducibility(self, seed: int = 0) -> None:
        """Limits sources of nondeterministic behavior.

        This function:
        * Seeds PyTorch, Python, and NumPy.
        * Disables CUDA convolution benchmarking.
        * Configures PyTorch to use determinstic algorithms.
        * Seeds workers spawned for multi-process data loading.

        Args:
            seed (int): The number to seed libraries and data workers with.
        """
        self._seed = seed

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


@PublicAPI(stability="beta")
@dataclass
class TorchConfig(BackendConfig):
    """Configuration for torch process group setup.

    See https://pytorch.org/docs/stable/distributed.html for more info.

    Args:
        backend (str): The backend to use for training.
            See ``torch.distributed.init_process_group`` for more info and
            valid values.
            If set to None, nccl will be used if GPUs are requested, else gloo
            will be used.
        init_method (str): The initialization method to use. Either "env"
            for environment variable initialization or "tcp" for TCP
            initialization. Defaults to "env".
        timeout_s (int): Seconds for process group operations to timeout.
    """

    backend: Optional[str] = None
    init_method: str = "env"
    timeout_s: int = 1800

    @property
    def backend_cls(self):
        return TorchBackend


def setup_torch_process_group(
    backend: str,
    world_rank: int,
    world_size: int,
    init_method: str,
    timeout_s: int = 1800,
):
    """Connects the distributed PyTorch backend.

    Args:
        backend (str): The backend (nccl, gloo, etc.) to use for training.
        world_rank (int): Rank of the current worker.
        world_size (int): Number of workers participating in the job.
        init_method (str): URL specifying how to initialize the process group.
        timeout_s (timedelta): Seconds for process group operations to timeout.
    """
    logger.info(
        f"Setting up process group for: {init_method} [rank={world_rank}, "
        f"world_size={world_size}]"
    )
    logger.debug(f"using {backend}")

    if backend == "nccl" and "NCCL_BLOCKING_WAIT" not in os.environ:
        logger.debug(
            "Setting NCCL_BLOCKING_WAIT for detecting node failure. "
            "To override this behavior, you can set NCCL_BLOCKING_WAIT=0."
        )
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=world_rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )


def shutdown_torch(destroy_process_group=False):
    if destroy_process_group:
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TorchBackend(Backend):
    share_cuda_visible_devices: bool = True

    def on_start(self, worker_group: WorkerGroup, backend_config: TorchConfig):
        if dist.is_available():
            # Set the appropriate training backend.
            if backend_config.backend is None:
                if worker_group.num_gpus_per_worker > 0:
                    backend = "nccl"
                else:
                    backend = "gloo"
            else:
                backend = backend_config.backend

            master_addr, master_port = worker_group.execute_single(
                0, get_address_and_port
            )
            if backend_config.init_method == "env":

                def set_env_vars(addr, port):
                    os.environ["MASTER_ADDR"] = addr
                    os.environ["MASTER_PORT"] = str(port)

                worker_group.execute(set_env_vars, addr=master_addr, port=master_port)
                url = "env://"
            elif backend_config.init_method == "tcp":
                url = f"tcp://{master_addr}:{master_port}"
            else:
                raise ValueError(
                    f"The provided init_method ("
                    f"{backend_config.init_method}) is not supported. Must "
                    f"be either 'env' or 'tcp'."
                )

            setup_futures = []
            for i in range(len(worker_group)):
                setup_futures.append(
                    worker_group.execute_single_async(
                        i,
                        setup_torch_process_group,
                        backend=backend,
                        world_rank=i,
                        world_size=len(worker_group),
                        init_method=url,
                        timeout_s=backend_config.timeout_s,
                    )
                )
            ray.get(setup_futures)
        else:
            raise RuntimeError("Distributed torch is not available.")

    def on_shutdown(self, worker_group: WorkerGroup, backend_config: TorchConfig):

        worker_group.execute(
            shutdown_torch, destroy_process_group=len(worker_group) > 1
        )

    @staticmethod
    def encode_data(data_dict: Dict) -> EncodedData:
        """Special handling for moving model from worker to driver."""

        # If model is being checkpointed and is wrapped in DDP, then extract
        # out the underlying module. If not, then deserialization will fail
        # since the torch process group is not initialized on the driver.

        for k, v in data_dict.items():
            if isinstance(v, DistributedDataParallel) and hasattr(v, "module"):
                data_dict[k] = v.module

        # Convert the checkpoint dict to bytes, so that any GPU tensors that
        # are in the checkpoint dict can be properly deserialized on the
        # driver side, even if the driver does not have access to a GPU device.
        _buffer = io.BytesIO()
        torch.save(data_dict, _buffer)
        return _buffer.getvalue()

    @staticmethod
    def decode_data(encoded_data: EncodedData) -> Dict:
        # When decoding the bytes on the driver side, always map to CPU.
        _buffer = io.BytesIO(encoded_data)
        checkpoint_dict = torch.load(_buffer, map_location="cpu")
        return checkpoint_dict


class _WrappedDataLoader(DataLoader):
    def __init__(self, base_dataloader: DataLoader, device: torch.device):

        self.__dict__.update(getattr(base_dataloader, "__dict__", {}))
        self.dataloader = base_dataloader
        self.device = device

    def _move_to_device(self, item):
        def try_move_device(i):
            try:
                i = i.to(self.device)
            except AttributeError:
                logger.debug(f"Item {i} cannot be moved to device " f"{self.device}.")
            return i

        return tuple(try_move_device(i) for i in item)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)

        for item in iterator:
            yield self._move_to_device(item)


@PublicAPI(stability="beta")
def get_device() -> torch.device:
    """Gets the correct torch device to use for training."""
    return get_accelerator(TorchAccelerator).get_device()


@PublicAPI(stability="beta")
def prepare_model(
    model: torch.nn.Module,
    move_to_device: bool = True,
    wrap_ddp: bool = True,
    ddp_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """Prepares the model for distributed execution.

    This allows you to use the same exact code regardless of number of
    workers or the device type being used (CPU, GPU).

    Args:
        model (torch.nn.Module): A torch model to prepare.
        move_to_device (bool): Whether to move the model to the correct
            device. If set to False, the model needs to manually be moved
            to the correct device.
        wrap_ddp (bool): Whether to wrap models in
            ``DistributedDataParallel``.
        ddp_kwargs (Dict[str, Any]): Args to pass into
            ``DistributedDataParallel`` initialization if ``wrap_ddp`` is
            set to True.
    """
    return get_accelerator(TorchAccelerator).prepare_model(
        model,
        move_to_device=move_to_device,
        wrap_ddp=wrap_ddp,
        ddp_kwargs=ddp_kwargs,
    )


@PublicAPI(stability="beta")
def prepare_data_loader(
    data_loader: torch.utils.data.DataLoader,
    add_dist_sampler: bool = True,
    move_to_device: bool = True,
) -> torch.utils.data.DataLoader:
    """Prepares DataLoader for distributed execution.

    This allows you to use the same exact code regardless of number of
    workers or the device type being used (CPU, GPU).

    Args:
        data_loader (torch.utils.data.DataLoader): The DataLoader to
            prepare.
        add_dist_sampler (bool): Whether to add a DistributedSampler to
            the provided DataLoader.
        move_to_device (bool): If set, automatically move the data
            returned by the data loader to the correct device.
    """
    return get_accelerator(TorchAccelerator).prepare_data_loader(
        data_loader,
        add_dist_sampler=add_dist_sampler,
        move_to_device=move_to_device,
    )


@PublicAPI(stability="beta")
def accelerate() -> None:
    """Enables training optimizations."""
    try:
        set_accelerator(TorchAccelerator())
    except RuntimeError:
        raise RuntimeError(
            "An accelerator has already been set. Make sure "
            "`train.torch.accelerate()` is not called multiple times, and is called "
            "before any of the prepare methods."
        )


@PublicAPI(stability="beta")
def enable_reproducibility(seed: int = 0) -> None:
    """Limits sources of nondeterministic behavior.

    This function:
    * Seeds PyTorch, Python, and NumPy.
    * Disables CUDA convolution benchmarking.
    * Configures PyTorch to use determinstic algorithms.
    * Seeds workers spawned for multi-process data loading.

    Args:
        seed (int): The number to seed libraries and data workers with.
    """
    get_accelerator(TorchAccelerator).enable_reproducibility(seed)


WORKER_TRACE_DIR_NAME = "pytorch_profiler_worker_traces"


class TorchWorkerProfiler:
    """Utility class for running PyTorch Profiler on a Train worker.

    Args:
        trace_dir (Optional[str]): The directory to store traces on the
           worker node. If ``None``, this will use a default temporary dir.
    """

    def __init__(self, trace_dir: Optional[str] = None):
        if profile is None:
            raise ImportError(
                "Torch Profiler requires torch>=1.8.1. "
                "Run `pip install 'torch>=1.8.1'` to use TorchWorkerProfiler."
            )

        trace_dir = trace_dir or Path(tempfile.gettempdir()).joinpath(
            WORKER_TRACE_DIR_NAME
        )
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        # Accumulated traces.
        self.profiler_trace_filenames = []

    def trace_handler(self, p: profile):
        """A stateful PyTorch Profiler trace handler.

        This will the export chrome trace to a file on disk.

        These exported traces can then be fetched by calling
        ``get_and_clear_profile_traces``.

        Args:
            p (profile): A PyTorch Profiler profile.
        """
        trace_filename = f"worker_{train.world_rank()}_epoch_{p.step_num}.pt.trace.json"
        trace_path = self.trace_dir.joinpath(trace_filename)

        logger.debug(f"Writing worker trace to {trace_path}.")
        p.export_chrome_trace(str(trace_path))
        self.profiler_trace_filenames.append(trace_filename)

    def get_and_clear_profile_traces(self):
        """Reads unread Profiler traces from this worker.

        Returns:
            The traces in a format consumable by
            ``TorchTensorboardProfilerCallback``.
        """

        def get_trace(filename):
            trace_path = self.trace_dir.joinpath(filename)
            return trace_path.read_text()

        traces = [
            (trace_filename, get_trace(trace_filename))
            for trace_filename in self.profiler_trace_filenames
        ]

        self.profiler_trace_files = []
        return {PYTORCH_PROFILER_KEY: traces}
