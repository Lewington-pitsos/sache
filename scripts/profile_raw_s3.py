import boto3
import torch
import warnings
import json
import time
import asyncio
import aiohttp
import randomname
import threading
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.train import SAE


async def request_chunk(session, url, start, end):
    headers = {
        "Range": f"bytes={start}-{end}",
    }
    async with session.get(url, headers=headers) as response:
        return start, await response.read()

async def download_chunks(session, url, total_size, chunk_size):
    chunks = [(i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)]

    tasks = [asyncio.create_task(request_chunk(session, url, start, end)) for start, end in chunks]
    return await asyncio.gather(*tasks)


KB = 1024
MB = KB * KB

# chunk_sizes = [KB * 512, MB * 2, MB * 8, MB * 32, MB * 128]
# thread_numbers = [4, 8, 16, 32, 64]
# total_sizes = [MB * 512, MB * 1024, MB * 2048, MB * 4096]

chunk_sizes = [MB * 16] * 8
concurrency = [128]
total_sizes = [5368709120]

async def main():
    run_name = randomname.generate('adj/', 'n/')

    print('run_name:', run_name)

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
   

    sae = SAE(n_features=768, hidden_size=1024, device='cuda')
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    cache = S3RCache(s3_client, 'tensors' 'lewington-pitsos-sache', chunk_size=MB * 16)
    iterator = iter(cache)
    total_size = cache.metadata['bytes_per_file']
    for i in range(8):
        start = time.time()

        t = next(iterator)


        for i in range(0, 1280, 64):
            optimizer.zero_grad()
            batch = t[i:i+64]

            reconstruction, _ = sae(batch)
            rmse = torch.sqrt(torch.mean((batch - reconstruction) ** 2))
            rmse.backward()
            optimizer.step()

        end = time.time()


        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

    cache.stop_downloading()


if __name__ == "__main__":
    asyncio.run(main())
