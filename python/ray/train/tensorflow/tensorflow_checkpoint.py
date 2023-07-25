import os
from typing import TYPE_CHECKING, Callable, Optional, Union

from enum import Enum
from os import path
import tensorflow as tf
from tensorflow import keras
import warnings

from ray.air._internal.checkpointing import save_preprocessor_to_dir
from ray.air.checkpoint import Checkpoint
from ray.air.constants import MODEL_KEY, PREPROCESSOR_KEY
from ray.train.data_parallel_trainer import _load_checkpoint_dict
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor


@PublicAPI(stability="beta")
class TensorflowCheckpoint(Checkpoint):
    """A :py:class:`~ray.air.checkpoint.Checkpoint` with TensorFlow-specific
    functionality.

    Create this from a generic :py:class:`~ray.air.checkpoint.Checkpoint` by calling
    ``TensorflowCheckpoint.from_checkpoint(ckpt)``.
    """

    _SERIALIZED_ATTRS = Checkpoint._SERIALIZED_ATTRS + ("_flavor", "_h5_file_path")

    class Flavor(Enum):
        # Various flavors with which TensorflowCheckpoint is generated.
        # This is necessary metadata to decide how to load model from a checkpoint.
        MODEL_WEIGHTS = 1
        SAVED_MODEL = 2
        H5 = 3

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._flavor = None
        # Will only be set when `self._flavor` is `H5`.
        self._h5_file_path = None

    @classmethod
    def from_model(
        cls,
        model: keras.Model,
        *,
        preprocessor: Optional["Preprocessor"] = None,
    ) -> "TensorflowCheckpoint":
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Keras
        model.

        The checkpoint created with this method needs to be paired with
        `model` when used.

        Args:
            model: The Keras model, whose weights are stored in the checkpoint.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :py:class:`TensorflowCheckpoint` containing the specified model.

        Examples:
        
            .. testcode::

                from ray.train.tensorflow import TensorflowCheckpoint
                import tensorflow as tf

                model = tf.keras.applications.resnet.ResNet101()
                checkpoint = TensorflowCheckpoint.from_model(model)

            .. testcode::
                :options: +MOCK
                :hide:

                ...  # Model may or may not be downloaded

        """
        checkpoint = cls.from_dict(
            {PREPROCESSOR_KEY: preprocessor, MODEL_KEY: model.get_weights()}
        )
        checkpoint._flavor = cls.Flavor.MODEL_WEIGHTS
        return checkpoint

    @classmethod
    def from_h5(
        cls, file_path: str, *, preprocessor: Optional["Preprocessor"] = None
    ) -> "TensorflowCheckpoint":
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Keras
        model from H5 format.

        The checkpoint generated by this method contains all the information needed.
        Thus no `model` is needed to be supplied when using this checkpoint.

        `file_path` must maintain validity even after this function returns.
        Some new files/directories may be added to the parent directory of `file_path`,
        as a side effect of this method.

        Args:
            file_path: The path to the .h5 file to load model from. This is the
                same path that is used for ``model.save(path)``.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :py:class:`TensorflowCheckpoint` converted from h5 format.

        Examples:

        .. testcode::

            import tensorflow as tf

            import ray
            from ray.train.batch_predictor import BatchPredictor
            from ray.train.tensorflow import (
                TensorflowCheckpoint, TensorflowTrainer, TensorflowPredictor
            )
            from ray.air import session
            from ray.air.config import ScalingConfig

            def train_func():
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=()),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(10),
                        tf.keras.layers.Dense(1),
                    ]
                )
                model.save("my_model.h5")
                checkpoint = TensorflowCheckpoint.from_h5("my_model.h5")
                session.report({"my_metric": 1}, checkpoint=checkpoint)

            trainer = TensorflowTrainer(
                train_loop_per_worker=train_func,
                scaling_config=ScalingConfig(num_workers=2))

            result_checkpoint = trainer.fit().checkpoint

            batch_predictor = BatchPredictor.from_checkpoint(
                result_checkpoint, TensorflowPredictor)
            batch_predictor.predict(ray.data.range(3))

        .. testoutput::
            :hide:

            ...

        """
        if not path.isfile(file_path) or not file_path.endswith(".h5"):
            raise ValueError(
                "Please supply a h5 file path to `TensorflowCheckpoint.from_h5()`."
            )
        dir_path = path.dirname(os.path.abspath(file_path))
        if preprocessor:
            save_preprocessor_to_dir(preprocessor, dir_path)
        checkpoint = cls.from_directory(dir_path)
        checkpoint._flavor = cls.Flavor.H5
        checkpoint._h5_file_path = os.path.basename(file_path)
        return checkpoint

    @classmethod
    def from_saved_model(
        cls, dir_path: str, *, preprocessor: Optional["Preprocessor"] = None
    ) -> "TensorflowCheckpoint":
        """Create a :py:class:`~ray.air.checkpoint.Checkpoint` that stores a Keras
        model from SavedModel format.

        The checkpoint generated by this method contains all the information needed.
        Thus no `model` is needed to be supplied when using this checkpoint.

        `dir_path` must maintain validity even after this function returns.
        Some new files/directories may be added to `dir_path`, as a side effect
        of this method.

        Args:
            dir_path: The directory containing the saved model. This is the same
                directory as used by ``model.save(dir_path)``.
            preprocessor: A fitted preprocessor to be applied before inference.

        Returns:
            A :py:class:`TensorflowCheckpoint` converted from SavedModel format.

        Examples:

        .. testcode::

            import tensorflow as tf

            import ray
            from ray.train.batch_predictor import BatchPredictor
            from ray.train.tensorflow import (
            TensorflowCheckpoint, TensorflowTrainer, TensorflowPredictor)
            from ray.air import session
            from ray.air.config import ScalingConfig

            def train_fn():
                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=()),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(10),
                        tf.keras.layers.Dense(1),
                    ])
                model.save("my_model")
                checkpoint = TensorflowCheckpoint.from_saved_model("my_model")
                session.report({"my_metric": 1}, checkpoint=checkpoint)

            trainer = TensorflowTrainer(
                train_loop_per_worker=train_fn,
                scaling_config=ScalingConfig(num_workers=2))

            result_checkpoint = trainer.fit().checkpoint

            batch_predictor = BatchPredictor.from_checkpoint(
                result_checkpoint, TensorflowPredictor)
            batch_predictor.predict(ray.data.range(3))

        .. testoutput::
            :hide:

            ...

        """
        if preprocessor:
            save_preprocessor_to_dir(preprocessor, dir_path)
        checkpoint = cls.from_directory(dir_path)
        checkpoint._flavor = cls.Flavor.SAVED_MODEL
        return checkpoint

    def get_model(
        self,
        model: Optional[Union[tf.keras.Model, Callable[[], tf.keras.Model]]] = None,
        model_definition: Optional[Callable[[], tf.keras.Model]] = None,
    ) -> tf.keras.Model:
        """Retrieve the model stored in this checkpoint.

        Args:
            model: This arg is expected only if the original checkpoint
                was created via `TensorflowCheckpoint.from_model`.
            model_definition: This parameter is deprecated. Use `model` instead.

        Returns:
            The Tensorflow Keras model stored in the checkpoint.
        """
        # TODO: Remove `model_definition` in 2.6.
        if model_definition is not None:
            raise DeprecationWarning(
                "The `model_definition` parameter is deprecated. Use the `model` "
                "parameter instead."
            )

        if model is not None and self._flavor is not self.Flavor.MODEL_WEIGHTS:
            warnings.warn(
                "TensorflowCheckpoint was created from "
                "TensorflowCheckpoint.from_saved_model` or "
                "`TensorflowCheckpoint.from_h5`, which already contains all the "
                "information needed. This means: "
                "If you are using BatchPredictor, you should do "
                "`BatchPredictor.from_checkpoint(checkpoint, TensorflowPredictor)`"
                " by removing kwargs `model=`. "
                "If you are using TensorflowPredictor directly, you should do "
                "`TensorflowPredictor.from_checkpoint(checkpoint)` by "
                "removing kwargs `model=`."
            )
        if self._flavor is self.Flavor.MODEL_WEIGHTS and model is None:
            raise ValueError(
                "Expecting `model` argument when checkpoint is "
                "saved through `TensorflowCheckpoint.from_model()`."
            )

        if self._flavor is self.Flavor.MODEL_WEIGHTS:
            model = self._get_model_from_weights(model)
        elif self._flavor is self.Flavor.H5:
            model = self._get_model_from_h5()
        elif self._flavor == self.Flavor.SAVED_MODEL:
            model = self._get_model_from_saved_model()
        else:
            raise RuntimeError(
                "Avoid directly using `from_dict` or "
                "`from_directory` directly. Make sure "
                "that the checkpoint was generated by "
                "`TensorflowCheckpoint.from_model`, "
                "`TensorflowCheckpoint.from_saved_model` or "
                "`TensorflowCheckpoint.from_h5`."
            )

        return model

    def _get_model_from_weights(
        self, model: Union[tf.keras.Model, Callable[[], tf.keras.Model]]
    ) -> tf.keras.Model:
        assert self._flavor is self.Flavor.MODEL_WEIGHTS
        assert model is not None

        if callable(model):
            model = model()

        model_weights, _ = _load_checkpoint_dict(self, "TensorflowTrainer")
        model.set_weights(model_weights)

        return model

    def _get_model_from_h5(self) -> tf.keras.Model:
        assert self._flavor is self.Flavor.H5

        with self.as_directory() as checkpoint_dir:
            return keras.models.load_model(
                os.path.join(checkpoint_dir, self._h5_file_path)
            )

    def _get_model_from_saved_model(self) -> tf.keras.Model:
        assert self._flavor is self.Flavor.SAVED_MODEL

        with self.as_directory() as checkpoint_dir:
            return keras.models.load_model(checkpoint_dir)
