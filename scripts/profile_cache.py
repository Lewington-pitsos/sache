import boto3
import json
import time
import asyncio
import sys 
import os
import multiprocessing


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.constants import *

def main():
    p1 = multiprocessing.Process(target=download)
    p2 = multiprocessing.Process(target=download)

    p1.start()
    p2.start()
    print('started')

    p1.join()
    p2.join()

    print('finished')

def download():
    asyncio.run(_download())

async def _download():
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    device = 'cuda'

    cache = S3RCache(s3_client, 'merciless-citadel', 'lewington-pitsos-sache', chunk_size=MB * 8, concurrency=400)
    
    iter(cache)
    
    total_size = cache.metadata['bytes_per_file']
    for i in range(8):
        start = time.time()

        t = next(cache).to(device)

        end = time.time()


        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

    cache.stop_downloading()


if __name__ == "__main__":
    main()
