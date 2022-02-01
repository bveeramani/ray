from dataclasses import dataclass
import io
import logging
import os

from datetime import timedelta
from typing import Optional, Dict, Any, Callable

import ray
from ray.train.backend import BackendConfig, Backend, EncodedData
from ray.train.worker_group import WorkerGroup
from ray.train.utils import get_address_and_port

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ray.util import PublicAPI

from .session import get_session
from .accelerators import Accelerator, TorchAccelerator

logger = logging.getLogger(__name__)


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

    @property
    def default_accelerator_factory(self) -> Callable[[], Accelerator]:
        return TorchAccelerator


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
    return get_session().accelerator.prepare_model(
        model, move_to_device=move_to_device, wrap_ddp=wrap_ddp, ddp_kwargs=ddp_kwargs
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
    return get_session().accelerator.prepare_data_loader(
        data_loader, add_dist_sampler=add_dist_sampler, move_to_device=move_to_device
    )


def get_device() -> torch.device:
    """Gets the correct torch device to use for training."""
    return get_session().accelerator.get_device()


@PublicAPI(stability="beta")
def accelerate() -> None:
    """Enables training optimizations."""
    get_session().accelerator = TorchAccelerator()
