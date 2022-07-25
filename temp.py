import ray
from ray import train
from ray.data.preprocessors.batch_mapper import BatchMapper
from ray.train.torch import TorchTrainer
from ray.air import session
from ray.air.config import ScalingConfig

def train_loop_per_worker():
    model = Net()
    for iter in range(100):
        # Trainer will automatically handle sharding.
        data_shard = session.get_dataset_shard().to_torch()
        model.train(data_shard)
    return model

train_dataset = ray.data.from_items(
    [{"x": x, "y": x + 1} for x in range(32)])
trainer = TorchTrainer(train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=2),
    datasets={"train": train_dataset})

print(repr(trainer))



from ray.data.preprocessors import Concatenator


# print(Concatenator())


# def f(x):
#     return x
# print(BatchMapper(lambda x: x))
