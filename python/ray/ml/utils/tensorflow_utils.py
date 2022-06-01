import numpy as np
import pandas as pd
import tensorflow as tf


def convert_pandas_to_tf_tensor(df: pd.DataFrame) -> tf.Tensor:
    """Convert a pandas dataframe to a TensorFlow tensor.

    This function works in two steps:
    1. Convert each dataframe column to a tensor.
    2. Concatenate the resulting tensors along the last axis.

    Arguments:
        df: The dataframe to convert to a TensorFlow tensor. Columns must be of
            a numeric dtype, ``TensorDtype``, or object dtype. If a column has
            an object dtype, the column must contain ``ndarray`` objects.

    Returns:
        A tensor constructed from the dataframe.

    Raises:
        ValueError: if a column has an invalid dtype.
        ValueError: if the columns can't be combined into a single tensor.

    Examples:
        >>> import pandas as pd
        >>> from ray.ml.utils.tensorflow_utils improt convert_pandas_to_tf_tensor
        >>>
        >>> df = pd.DataFrame({"X1": [1, 2, 3], "X2": [4, 5, 6]})
        >>> convert_pandas_to_tf_tensor(df[["X1"]]).shape
        TensorShape([3, 1])
        >>> convert_pandas_to_tf_tensor(df[["X1", "X2"]]).shape
        TensorShape([3, 2])

        >>> from ray.data.extensions import TensorArray
        >>>
        >>> df = pd.DataFrame({"image": TensorArray(np.zeros((4, 3, 32, 32)))})
        >>> convert_pandas_to_tf_tensor(df).shape
        TensorShape([4, 3, 32, 32])
    """

    def tensorize(series):
        try:
            return tf.convert_to_tensor(series)
        except ValueError:
            return tf.stack([tensorize(element) for element in series])

    try:
        # We need to cast the tensors to a common type so that we can concatenate them.
        # If the columns contain different types (for example, `float32`s and
        # `int32`s), then `tf.expand_dims` raises an error.
        common_type: np.dtype = np.find_common_type(df.dtypes, [])
        df = df.astype(common_type)
    except TypeError:
        # `find_common_type` fails if a series has `TensorDtype`. In this case,
        # don't cast any of the series and continue.
        pass

    tensors = []
    for column in df.columns:
        series = df[column]
        tensor = tensorize(series)
        tensors.append(tensor)

    if len(tensors) > 1:
        tensors = [tf.expand_dims(tensor, axis=-1) for tensor in tensors]

    concatenated_tensor = tf.concat(tensors, axis=-1)

    if concatenated_tensor.ndim == 1:
        return tf.expand_dims(concatenated_tensor, axis=1)

    return concatenated_tensor
