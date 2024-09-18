from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

output_dir = os.path.abspath("../laion_imgs")

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

spark = (
    SparkSession.builder.config("spark.driver.memory", "32G").master("local[16]").appName("spark-stats").getOrCreate()
)

download(
    processes_count=12,
    thread_count=32,
    url_list="../laion/pq-10/",
    image_size=224, # as used by https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/tree/main
    output_folder=output_dir,
    output_format="webdataset",
    input_format="parquet",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=True,
    number_sample_per_shard=1000,
    distributor="pyspark",
)
