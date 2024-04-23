import io
from typing import Iterable, Optional, Tuple

import numpy as np
import pyarrow

from ray.data._internal.util import _check_import
from ray.data.block import DataBatch
from ray.data.datasource.file_reader import FileReader


class ImageReader(FileReader):

    NUM_THREADS = 8

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
    ):
        _check_import(self, module="PIL", package="Pillow")

        if size is not None and len(size) != 2:
            raise ValueError(
                "Expected `size` to contain two integers for height and width, "
                f"but got {len(size)} integers instead."
            )

        if size is not None and (size[0] < 0 or size[1] < 0):
            raise ValueError(
                f"Expected `size` to contain positive integers, but got {size} instead."
            )

        self.size = size
        self.mode = mode

    def read_stream(self, file: "pyarrow.NativeFile", path: str) -> Iterable[DataBatch]:
        from PIL import Image, UnidentifiedImageError

        data = file.readall()

        try:
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as e:
            raise ValueError(f"PIL couldn't load image file at path '{path}'.") from e

        if self.size is not None:
            height, width = self.size
            image = image.resize((width, height), resample=Image.BILINEAR)
        if self.mode is not None:
            image = image.convert(self.mode)

        batch = np.expand_dims(np.array(image), axis=0)
        yield {"image": batch}
