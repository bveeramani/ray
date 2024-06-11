import abc
from typing import TYPE_CHECKING, List, Union

from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.util import unify_block_metadata_schema
from ray.data.block import Block, BlockMetadata
from ray.types import ObjectRef

if TYPE_CHECKING:
    import pyarrow as pa

    ArrowTable = Union["pa.Table", bytes]


class AbstractFrom(LogicalOperator, metaclass=abc.ABCMeta):
    """Abstract logical operator for `from_*`."""

    def __init__(
        self,
        input_blocks: List[ObjectRef[Block]],
        input_metadata: List[BlockMetadata],
    ):
        super().__init__(self.__class__.__name__, [], len(input_blocks))
        assert len(input_blocks) == len(input_metadata), (
            len(input_blocks),
            len(input_metadata),
        )
        # `owns_blocks` is False because this op may be shared by multiple Datasets.
        self._input_data = [
            RefBundle([(input_blocks[i], input_metadata[i])], owns_blocks=False)
            for i in range(len(input_blocks))
        ]

    @property
    def input_data(self) -> List[RefBundle]:
        return self._input_data

    def schema(self):
        metadata = [m for bundle in self._input_data for m in bundle.metadata]
        return unify_block_metadata_schema(metadata)

    def num_rows(self):
        if all(bundle.num_rows() is not None for bundle in self._input_data):
            return sum(bundle.num_rows() for bundle in self._input_data)
        else:
            return None

    def num_blocks(self):
        if self._input_data is None:
            return None
        return sum(len(bundle.blocks) for bundle in self._input_data)

    def output_data(self):
        blocks = []
        for bundle in self._input_data:
            blocks.extend(bundle.blocks)
        return RefBundle(blocks, owns_blocks=False)


class FromItems(AbstractFrom):
    """Logical operator for `from_items`."""

    pass


class FromBlocks(AbstractFrom):
    """Logical operator for `from_blocks`."""

    pass


class FromNumpy(AbstractFrom):
    """Logical operator for `from_numpy`."""

    pass


class FromArrow(AbstractFrom):
    """Logical operator for `from_arrow`."""

    pass


class FromPandas(AbstractFrom):
    """Logical operator for `from_pandas`."""

    pass
