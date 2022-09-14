from typing import Optional


def PublicAPI(*args, **kwargs):
    """Annotation for documenting public APIs.

    Public APIs are classes and methods exposed to end users of Ray.

    If ``stability="alpha"``, the API can be used by advanced users who are
    tolerant to and expect breaking changes.

    If ``stability="beta"``, the API is still public and can be used by early
    users, but are subject to change.

    If ``stability="stable"``, the APIs will remain backwards compatible across
    minor Ray releases (e.g., Ray 1.4 -> 1.8).

    For a full definition of the stability levels, please refer to the
    :ref:`Ray API Stability definitions <api-stability>`.

    Args:
        stability: One of {"stable", "beta", "alpha"}.

    Examples:
        >>> from ray.util.annotations import PublicAPI
        >>> @PublicAPI
        ... def func(x):
        ...     return x

        >>> @PublicAPI(stability="beta")
        ... def func(y):
        ...     return y
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return PublicAPI(stability="stable")(args[0])

    if "stability" in kwargs:
        stability = kwargs["stability"]
        assert stability in ["stable", "beta", "alpha"], stability
    elif kwargs:
        raise ValueError("Unknown kwargs: {}".format(kwargs.keys()))
    else:
        stability = "stable"

    def wrap(obj):
        if not obj.__doc__:
            obj.__doc__ = ""
        if stability in ["alpha", "beta"]:
            obj.__doc__ += (
                f"\n    PublicAPI ({stability}): This API is in {stability} "
                "and may change before becoming stable."
            )
        else:
            obj.__doc__ += "\n    PublicAPI: This API is stable across Ray releases."

        _mark_annotated(obj)
        return obj

    return wrap


def DeveloperAPI(*args, **kwargs):
    """Annotation for documenting developer APIs.

    Developer APIs are lower-level methods explicitly exposed to advanced Ray
    users and library developers. Their interfaces may change across minor
    Ray releases.

    Examples:
        >>> from ray.util.annotations import DeveloperAPI
        >>> @DeveloperAPI
        ... def func(x):
        ...     return x
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return DeveloperAPI()(args[0])

    def wrap(obj):
        if not obj.__doc__:
            obj.__doc__ = ""
        obj.__doc__ += (
            "\n    DeveloperAPI: This API may change across minor Ray releases."
        )
        _mark_annotated(obj)
        return obj

    return wrap


def Deprecated(*args, **kwargs):
    """Annotation for documenting a deprecated API.

    Deprecated APIs may be removed in future releases of Ray.

    Args:
        message: a message to help users understand the reason for the
            deprecation, and provide a migration path.

    Examples:
        >>> from ray.util.annotations import Deprecated
        >>> @Deprecated
        ... def func(x):
        ...     return x

        >>> @Deprecated(message="g() is deprecated because the API is error "
        ...   "prone. Please call h() instead.")
        ... def g(y):
        ...     return y
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Deprecated()(args[0])

    deprecation_message = (
        "**DEPRECATED:** This API is deprecated and may be removed in a future Ray"
        "release."
    )

    additional_info = None
    if "message" in kwargs:
        additional_info = kwargs["message"]
        del kwargs["message"]

    if kwargs:
        raise ValueError("Unknown kwargs: {}".format(kwargs.keys()))

    def inner(obj):
        if not obj.__doc__:
            obj.__doc__ = ""

        obj.__doc__ = obj.__doc__.rstrip()

        indent = _get_indent(obj.__doc__)
        obj.__doc__ += "\n\n"
        obj.__doc__ += f"{' ' * indent}.. warning::\n"
        obj.__doc__ += f"{' ' * (indent + 4)}{deprecation_message}"
        if additional_info:
            obj.__doc__ += f" {additional_info}"
        obj.__doc__ += f"\n{' ' * indent}"

        _mark_annotated(obj)
        return obj

    return inner


def _get_indent(docstring: Optional[str]) -> int:
    """

    Example:
        >>> def f():
        ...     '''Docstring summary.'''
        >>> f.__doc__
        'Docstring summary.'
        >>> _get_indent(f.__doc__)
        0

        >>> def g(foo):
        ...     '''Docstring summary.
        ...
        ...     Args:
        ...         foo: Does bar.
        ...     '''
        >>> g.__doc__
        'Docstring summary.\\n\\n    Args:\\n        foo: Does bar.\\n    '
        >>> _get_indent(g.__doc__)
        4

        >>> class A:
        ...     def h():
        ...         '''Docstring summary.
        ...
        ...         Returns:
        ...             None.
        ...         '''
        >>> A.h.__doc__
        'Docstring summary.\\n\\n        Returns:\\n            None.\\n        '
        >>> _get_indent(A.h.__doc__)
        8
    """
    if not docstring:
        return 0

    non_empty_lines = list(filter(bool, docstring.splitlines()))
    if len(non_empty_lines) == 1:
        # Docstring contains summary only.
        return 0

    # The docstring summary isn't indented, so check the indentation of the second
    # non-empty line.
    return len(non_empty_lines[1]) - len(non_empty_lines[1].lstrip())


def _mark_annotated(obj) -> None:
    # Set magic token for check_api_annotations linter.
    if hasattr(obj, "__name__"):
        obj._annotated = obj.__name__


def _is_annotated(obj) -> bool:
    # Check the magic token exists and applies to this class (not a subclass).
    return hasattr(obj, "_annotated") and obj._annotated == obj.__name__
