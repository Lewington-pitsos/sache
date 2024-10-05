from img2dataset import download
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

if __name__ == "__main__":
    spark = (
        SparkSession.builder.config("spark.driver.memory", "32G").master("local[16]").appName("spark-stats").getOrCreate()
    )

    download(
        processes_count=14,
        thread_count=64,
        output_folder="laion/images",
        url_list="cruft/pq",
        image_size=224, 
        output_format="files",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        distributor="pyspark",
        min_image_size=64,
    )
