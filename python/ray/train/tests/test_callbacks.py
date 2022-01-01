import pytest
import os
import shutil
import tempfile
import json
import glob
from collections import defaultdict

import ray
import ray.train as train
from ray.train import Trainer
from ray.train.constants import (
    TRAINING_ITERATION,
    DETAILED_AUTOFILLED_KEYS,
    BASIC_AUTOFILLED_KEYS,
    ENABLE_DETAILED_AUTOFILLED_METRICS_ENV,
)
from ray.train.callbacks import JsonLoggerCallback, TBXLoggerCallback
from ray.train.backend import BackendConfig, Backend
from ray.train.worker_group import WorkerGroup
from ray.train.callbacks.logging import MLflowLoggerCallback

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
except ImportError:
    summary_iterator = None


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def make_temp_dir():
    tmpdir = str(tempfile.mkdtemp())
    yield tmpdir
    # The code after the yield will run as teardown code.
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


class TestConfig(BackendConfig):
    @property
    def backend_cls(self):
        return TestBackend


class TestBackend(Backend):
    def on_start(self, worker_group: WorkerGroup, backend_config: TestConfig):
        pass

    def on_shutdown(self, worker_group: WorkerGroup, backend_config: TestConfig):
        pass


@pytest.mark.parametrize("workers_to_log", [0, None, [0, 1]])
@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("filename", [None, "my_own_filename.json"])
def test_json(ray_start_4_cpus, make_temp_dir, workers_to_log, detailed, filename):
    if detailed:
        os.environ[ENABLE_DETAILED_AUTOFILLED_METRICS_ENV] = "1"

    config = TestConfig()

    num_iters = 5
    num_workers = 4

    if workers_to_log is None:
        num_workers_to_log = num_workers
    elif isinstance(workers_to_log, int):
        num_workers_to_log = 1
    else:
        num_workers_to_log = len(workers_to_log)

    def train_func():
        for i in range(num_iters):
            train.report(index=i)
        return 1

    if filename is None:
        # if None, use default value
        callback = JsonLoggerCallback(workers_to_log=workers_to_log)
    else:
        callback = JsonLoggerCallback(filename=filename, workers_to_log=workers_to_log)
    trainer = Trainer(config, num_workers=num_workers, logdir=make_temp_dir)
    trainer.start()
    trainer.run(train_func, callbacks=[callback])
    if filename is None:
        assert str(callback.log_path.name) == JsonLoggerCallback._default_filename
    else:
        assert str(callback.log_path.name) == filename

    with open(callback.log_path, "r") as f:
        log = json.load(f)
    print(log)
    assert len(log) == num_iters
    assert len(log[0]) == num_workers_to_log
    assert all(len(element) == len(log[0]) for element in log)
    assert all(
        all(worker["index"] == worker[TRAINING_ITERATION] - 1 for worker in element)
        for element in log
    )
    assert all(
        all(all(key in worker for key in BASIC_AUTOFILLED_KEYS) for worker in element)
        for element in log
    )
    if detailed:
        assert all(
            all(
                all(key in worker for key in DETAILED_AUTOFILLED_KEYS)
                for worker in element
            )
            for element in log
        )
    else:
        assert all(
            all(
                not any(key in worker for key in DETAILED_AUTOFILLED_KEYS)
                for worker in element
            )
            for element in log
        )

    os.environ.pop(ENABLE_DETAILED_AUTOFILLED_METRICS_ENV, 0)
    assert ENABLE_DETAILED_AUTOFILLED_METRICS_ENV not in os.environ


def _validate_tbx_result(events_dir):
    events_file = list(glob.glob(f"{events_dir}/events*"))[0]
    results = defaultdict(list)
    for event in summary_iterator(events_file):
        for v in event.summary.value:
            assert v.tag.startswith("ray/train")
            results[v.tag[10:]].append(v.simple_value)

    assert len(results["episode_reward_mean"]) == 3
    assert [int(res) for res in results["episode_reward_mean"]] == [4, 5, 6]
    assert len(results["score"]) == 1
    assert len(results["hello/world"]) == 1


def test_TBX(ray_start_4_cpus, make_temp_dir):
    config = TestConfig()

    temp_dir = make_temp_dir
    num_workers = 4

    def train_func():
        train.report(episode_reward_mean=4)
        train.report(episode_reward_mean=5)
        train.report(episode_reward_mean=6, score=[1, 2, 3], hello={"world": 1})
        return 1

    callback = TBXLoggerCallback(temp_dir)
    trainer = Trainer(config, num_workers=num_workers)
    trainer.start()
    trainer.run(train_func, callbacks=[callback])

    _validate_tbx_result(temp_dir)


def test_mlflow(ray_start_4_cpus, make_temp_dir):
    config = TestConfig()

    params = {"p1": "p1"}

    temp_dir = make_temp_dir
    num_workers = 4

    def train_func(config):
        train.report(episode_reward_mean=4)
        train.report(episode_reward_mean=5)
        train.report(episode_reward_mean=6)
        return 1

    callback = MLflowLoggerCallback(experiment_name="test_exp", logdir=temp_dir)
    trainer = Trainer(config, num_workers=num_workers)
    trainer.start()
    trainer.run(train_func, config=params, callbacks=[callback])

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=callback.mlflow_util._mlflow.get_tracking_uri())

    all_runs = callback.mlflow_util._mlflow.search_runs(experiment_ids=["0"])
    assert len(all_runs) == 1
    # all_runs is a pandas dataframe.
    all_runs = all_runs.to_dict(orient="records")
    run_id = all_runs[0]["run_id"]
    run = client.get_run(run_id)

    assert run.data.params == params
    assert (
        "episode_reward_mean" in run.data.metrics
        and run.data.metrics["episode_reward_mean"] == 6.0
    )
    assert (
        TRAINING_ITERATION in run.data.metrics
        and run.data.metrics[TRAINING_ITERATION] == 3.0
    )

    metric_history = client.get_metric_history(run_id=run_id, key="episode_reward_mean")

    assert len(metric_history) == 3
    iterations = [metric.step for metric in metric_history]
    assert iterations == [1, 2, 3]
    rewards = [metric.value for metric in metric_history]
    assert rewards == [4, 5, 6]


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
