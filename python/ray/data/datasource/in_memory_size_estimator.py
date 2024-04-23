from typing import List

from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockAccessor
from ray.data.datasource.file_reader import FileReader


class InMemorySizeEstimator:
    def estimate_in_memory_size(self, file_size: int) -> int:
        raise NotImplementedError


class SamplingInMemorySizeEstimator(InMemorySizeEstimator):
    def __init__(
        self, paths: List[str], filesystem: "pyarrow.fs.FileSystem", reader: FileReader
    ):
        from ray.data._internal.planner.plan_read_files_op import _get_file_infos

        file_path, file_size = next(
            _get_file_infos(paths[0], filesystem, ignore_missing_path=False)
        )
        batch = next(reader.read_paths([file_path], filesystem))

        builder = DelegatingBlockBuilder()
        builder.add_batch(batch)
        block = builder.build()

        in_memory_size = BlockAccessor.for_block(block).size_bytes()
        self._encoding_ratio = in_memory_size / file_size

    def estimate_in_memory_size(self, file_size: int) -> int:
        return file_size * self._encoding_ratio
