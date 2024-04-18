import logging
from timeit import default_timer as timer
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

import ray
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    BuildOutputBlocksMapTransformFn,
    MapTransformer,
    MapTransformFn,
)
from ray.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_meta_provider import _handle_read_os_error

logger = logging.getLogger(__name__)


def _get_file_infos(
    path: str,
    filesystem: "pyarrow.fs.FileSystem",
    ignore_missing_path: bool,
) -> Iterable[Tuple[str, int]]:
    from pyarrow.fs import FileType

    try:
        file_info = filesystem.get_file_info(path)
    except OSError as e:
        _handle_read_os_error(e, path)

    if file_info.type == FileType.Directory:
        yield from _expand_directory(path, filesystem, ignore_missing_path)
    elif file_info.type == FileType.File:
        yield (path, file_info.size)
    elif file_info.type == FileType.NotFound and ignore_missing_path:
        pass
    else:
        raise FileNotFoundError(path)


def _expand_directory(
    path: str,
    filesystem: "pyarrow.fs.FileSystem",
    ignore_missing_path: bool,
) -> Iterable[Tuple[str, int]]:
    exclude_prefixes = [".", "_"]

    from pyarrow.fs import FileSelector

    selector = FileSelector(path, recursive=True, allow_not_found=ignore_missing_path)
    files = filesystem.get_file_info(selector)
    base_path = selector.base_dir
    for file_ in files:
        if not file_.is_file:
            continue
        file_path = file_.path
        if not file_path.startswith(base_path):
            continue
        relative = file_path[len(base_path) :]
        if any(relative.startswith(prefix) for prefix in exclude_prefixes):
            continue
        yield (file_path, file_.size)


def create_input_data_buffer(paths: List[str]) -> InputDataBuffer:
    path_splits = np.array_split(paths, min(200, len(paths)))  # TODO: 200 is arbitrary.

    input_data = []
    for path_split in path_splits:
        block = pa.Table.from_pydict({"path": path_split})
        metadata = BlockAccessor.for_block(block).get_metadata(
            input_files=None, exec_stats=None
        )
        ref_bundle = RefBundle(
            [(ray.put(block), metadata)],
            # `owns_blocks` is False, because these refs are the root of the
            # DAG. We shouldn't eagerly free them. Otherwise, the DAG cannot
            # be reconstructed.
            owns_blocks=False,
        )
        input_data.append(ref_bundle)
    return InputDataBuffer(input_data=input_data)


def create_fetch_metadata_operator(
    input_op: PhysicalOperator, filesystem
) -> PhysicalOperator:
    def fetch_metadata(blocks: Iterable[Block], _: TaskContext) -> Iterable[Block]:
        running_file_size = 0
        file_paths = []
        file_sizes = []
        for block in blocks:
            assert isinstance(block, pa.Table)
            assert "path" in block.column_names, block.column_names
            for path in map(str, list(block["path"])):
                for file_path, file_size in _get_file_infos(path, filesystem, False):
                    file_paths.append(file_path)
                    file_sizes.append(file_size * 734.78)
                    # TODO: Estimate file size from metadata provider.
                    running_file_size += file_size * 734.78
                    if running_file_size > 128 * 1024 * 1024:
                        block = pa.Table.from_pydict(
                            {"path": file_paths, "size": file_sizes}
                        )
                        yield block
                        file_paths = []
                        file_sizes = []
                        running_file_size = 0

        if file_paths:
            assert file_sizes
            block = pa.Table.from_pydict({"path": file_paths, "size": file_sizes})
            yield block

    transform_fns: List[MapTransformFn] = [BlockMapTransformFn(fetch_metadata)]
    map_transformer = MapTransformer(transform_fns)
    return MapOperator.create(
        map_transformer,
        input_op,
        name="FetchMetadata",
        target_max_block_size=None,
    )


def create_read_files_operator(
    input_op: PhysicalOperator, filesystem
) -> PhysicalOperator:
    from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

    def read_path(blocks: Iterable[Block], _: TaskContext) -> Iterable[Block]:
        blocks = list(blocks)
        assert len(blocks) == 1, len(blocks)

        start_time = timer()
        last_yield_time = timer()

        for block in blocks:
            paths = map(str, list(block["path"]))
            for path in paths:
                import io

                from PIL import Image

                data = filesystem.open_input_stream(path).read()
                image = Image.open(io.BytesIO(data))

                # QUESTION: How do we do sampling for metadata esitation?
                # QUESTION: Why do we do metadata size estimation?
                # TODO: Replace with FileReader
                builder = DelegatingBlockBuilder()
                array = np.array(image)
                item = {"image": array}
                builder.add(item)
                yield builder.build()
                last_yield_time = timer()

    transform_fns: List[MapTransformFn] = [
        BlockMapTransformFn(read_path),
        BuildOutputBlocksMapTransformFn.for_blocks(),
    ]
    map_transformer = MapTransformer(transform_fns)
    return MapOperator.create(
        map_transformer,
        input_op,
        name="ReadImages",
        target_max_block_size=None,
    )


def plan_read_files_op(op: ReadFiles) -> PhysicalOperator:
    from ray.data.datasource.path_util import _resolve_paths_and_filesystem

    logger.info("Creating operators...")
    paths, filesystem = _resolve_paths_and_filesystem(op._paths, None)
    input_data_buffer = create_input_data_buffer(paths)
    fetch_metadata_operator = create_fetch_metadata_operator(
        input_data_buffer, filesystem
    )
    read_files_operator = create_read_files_operator(
        fetch_metadata_operator, filesystem
    )
    logger.info("Done creating operators...")
    return read_files_operator
