from enum import Enum
from inspect import signature, Parameter
from typing import Any


class ParameterValue(Enum):
    UNKNOWN = 0


class ReprMixin:

    @property
    def _parameters(self) -> list[Parameter]:
        return list(signature(self.__init__)._parameters.values())

    @property
    def _values(self) -> dict[Parameter, Any]:
        out = {}
        for parameter in self._parameters:
            try:
                out[parameter] = getattr(self, parameter.name)
            except AttributeError:
                out[parameter] = getattr(self, "_" + parameter.name, ParameterValue.UNKNOWN)
        return out

    @property
    def _default_values(self) -> dict[Parameter, Any]:
        out = {}
        for parameter in self._parameters:
            if parameter.default is not Parameter.empty:
                out[parameter] = parameter.default
        return out

    @property
    def _user_specified_values(self) -> dict[Parameter, Any]:
        out = {}
        for parameter in self._parameters:
            if parameter not in self._default_values or self._values[parameter] != self._default_values[parameter]:
                out[parameter] = self._values[parameter]
        return out

    def __repr__(self):
        arguments = []
        for parameter, value in self._values.items():
            if value is ParameterValue.UNKNOWN:
                value = "?"

            if callable(value):
                value = getattr(value, "__qualname__", "__name__")

            arguments.append(f"{parameter.name}={value}")

        return f"{self.__class__.__name__}({', '.join(arguments)})"
