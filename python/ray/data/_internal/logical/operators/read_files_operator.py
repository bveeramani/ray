from typing import List, Union

from ray.data._internal.logical.interfaces import LogicalOperator


class ReadFiles(LogicalOperator):
    def __init__(self, paths: Union[str, List[str]]):
        super().__init__(name="ReadFiles", input_dependencies=[])

        if isinstance(paths, str):
            paths = [paths]

        self._paths = paths
