from typing import List, Union

from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data.datasource.file_reader import FileReader


class ReadFiles(LogicalOperator):
    def __init__(
        self, paths: Union[str, List[str]], reader: FileReader, estimator, filesystem
    ):
        super().__init__(name="ReadFiles", input_dependencies=[])

        if isinstance(paths, str):
            paths = [paths]

        self._paths = paths
        self._reader = reader
        self._estimator = estimator
        self._filesystem = filesystem
