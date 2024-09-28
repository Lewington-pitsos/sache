from img2dataset import download
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

if __name__ == "__main__":
    spark = (
        SparkSession.builder.config("spark.driver.memory", "32G").master("local[16]").appName("spark-stats").getOrCreate()
    )

    download(
        processes_count=14,
        thread_count=64,
        output_folder="cruft/pq",
        url_list="laion/pq-1/",
        image_size=256, 
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        distributor="pyspark",
        min_image_size=64,
    )
