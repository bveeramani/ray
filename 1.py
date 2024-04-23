from timeit import default_timer as timer

import ray

start_time = timer()

ds = ray.data.read_images(["s3://anonymous@33856-empty-images"] * 10)
ds.take(5)
# for _ in ds.iter_batches(batch_size=None, batch_format="pyarrow"):
#     pass
print(timer() - start_time)
