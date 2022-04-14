import logging
from typing import TYPE_CHECKING, Callable, List

from ray.data.block import Block, BlockMetadata, T
from ray.data.datasource import Datasource, ReadTask

if TYPE_CHECKING:
    import torch.utils.data

logger = logging.getLogger(__name__)


class TorchDatasource(Datasource[T]):
    """A datasource that let's you use Torch datasets with Ray Data.

    .. warning::
        ``TorchDatasource`` doesn't support parallel reads. You should only use this
        datasource for small datasets like MNIST or CIFAR.

    Example:
        >>> dataset_factory = lambda: torchvision.datasets.MNIST("data", download=True)
        >>> dataset = ray.data.read_datasource(
                TorchDatasource(), parallelism=1, dataset_factory=dataset_factory
            )
        >>> next(dataset.iter_batches(batch_size=2))
        [(<PIL.Image.Image image mode=L size=28x28 at 0x1142CCA60>, 5)]
    """

    def prepare_read(
        self, parallelism: int, dataset_factory: Callable[[], torch.utils.data.Dataset]
    ) -> List[ReadTask]:
        """Return a read task that loads a Torch dataset.

        Arguments:
            parallelism: This argument isn't used.
            dataset_factory: A no-argument function that returns a Torch dataset.
        """
        import torch.utils.data

        if isinstance(dataset_factory, torch.utils.data.Dataset):
            raise ValueError(
                "Expected a function that returns a Torch dataset, but got a "
                "`torch.utils.data.Dataset` instead."
            )

        if parallelism > 1:
            logger.warn(
                "`TorchDatasource` doesn't support parallel reads. The `parallelism` "
                "argument will be ignored."
            )

        def read_fn() -> List[Block]:
            # Load the entire dataset into memory.
            block = list(dataset_factory())
            # Store the data in a single block.
            return [block]

        metadata = BlockMetadata(
            num_rows=None,
            size_bytes=None,
            schema=None,
            input_files=None,
            exec_stats=None,
        )
        return [ReadTask(read_fn, metadata)]
