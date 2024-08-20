import boto3
import json
import time
import asyncio
import sys 
import os
import multiprocessing as mp


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.constants import *

def main():
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

    cache = S3RCache(s3_client, 'merciless-citadel', 'lewington-pitsos-sache', chunk_size=MB * 8, concurrency=500, n_workers=3)
    
    iter(cache)
    
    total_size = cache.metadata['bytes_per_file']
    overall_start = time.time()
    n = 6
    for i in range(n):
        start = time.time()

        t = next(cache)

        end = time.time()


        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size * n / MB) / (overall_end - overall_start):.2f}")

    cache._stop_downloading()


if __name__ == "__main__":
    main()
