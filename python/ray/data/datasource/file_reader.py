import io
from typing import Iterable, List

import pyarrow

from ray.data._internal.util import make_async_gen
from ray.data.block import DataBatch
from ray.data.context import DataContext


class FileReader:
    NUM_THREADS = 0

    def open_input_source(
        self,
        filesystem: "pyarrow.fs.FileSystem",
        path: str,
        **open_args,
    ) -> "pyarrow.io.InputStream":
        """Opens a source path for reading and returns the associated Arrow NativeFile.

        The default implementation opens the source path as a sequential input stream,
        using ctx.streaming_read_buffer_size as the buffer size if none is given by the
        caller.

        Implementations that do not support streaming reads (e.g. that require random
        access) should override this method.
        """
        import pyarrow as pa
        from pyarrow.fs import HadoopFileSystem

        compression = open_args.get("compression", None)
        if compression is None:
            try:
                # If no compression manually given, try to detect
                # compression codec from path.
                compression = pa.Codec.detect(path).name
            except (ValueError, TypeError):
                # Arrow's compression inference on the file path
                # doesn't work for Snappy, so we double-check ourselves.
                import pathlib

                suffix = pathlib.Path(path).suffix
                if suffix and suffix[1:] == "snappy":
                    compression = "snappy"
                else:
                    compression = None

        buffer_size = open_args.pop("buffer_size", None)
        if buffer_size is None:
            ctx = DataContext.get_current()
            buffer_size = ctx.streaming_read_buffer_size

        if compression == "snappy":
            # Arrow doesn't support streaming Snappy decompression since the canonical
            # C++ Snappy library doesn't natively support streaming decompression. We
            # works around this by manually decompressing the file with python-snappy.
            open_args["compression"] = None
        else:
            open_args["compression"] = compression

        file = filesystem.open_input_stream(path, buffer_size=buffer_size, **open_args)

        if compression == "snappy":
            import snappy

            stream = io.BytesIO()
            if isinstance(filesystem, HadoopFileSystem):
                snappy.hadoop_snappy.stream_decompress(src=file, dst=stream)
            else:
                snappy.stream_decompress(src=file, dst=stream)
            stream.seek(0)

            file = pa.PythonFile(stream, mode="r")

        return file

    def read_stream(self, file: "pyarrow.NativeFile", path: str) -> Iterable[DataBatch]:
        raise NotImplementedError

    def read_paths(self, paths: List[str], filesystem) -> Iterable[DataBatch]:
        num_threads = self.NUM_THREADS
        if len(paths) < num_threads:
            num_threads = len(paths)

        def _read_paths(paths: List[str]):
            for path in paths:
                file = self.open_input_source(filesystem, path)
                yield from self.read_stream(file, path)

        if num_threads > 0:
            yield from make_async_gen(
                iter(paths),
                _read_paths,
                num_workers=num_threads,
            )
        else:
            yield from _read_paths(paths)
