from tqdm import tqdm

import ray

ds = ray.data.read_images_fast(
    ["s3://anonymous@33856-empty-images"], override_num_blocks=200
)
ds.take(5)
