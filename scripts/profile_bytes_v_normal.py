import random
import requests
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
import boto3
import torch
import aiohttp
import asyncio

full_start = time.time()

def generate_random_tensor(size_mb):
    tensor_size = round((size_mb * 1024**2)) // 4  # float32 has 4 bytes
    tensor = torch.randn(tensor_size // (1024), 1024)
    return tensor

with open('.credentials.json') as f:
    credentials = json.load(f)
s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

tensor = generate_random_tensor(1 * 5000)

tensor_bytes = tensor.numpy().tobytes()

# buffer = io.BytesIO()
# torch.save(tensor, buffer)
# buffer.seek(0)

i = random.randint(0, 100000)
key = f'tensors/profile-{i}.pt'
s3_client.put_object(
    Bucket='lewington-pitsos-sache', 
    Key=key, 
    Body=tensor_bytes,
    ContentLength=len(tensor_bytes), 
    ContentType='application/octet-stream'
)

elapsed = time.time() - full_start
print(f'finished uploading, seconds elapsed {elapsed:.2f}')

url = f'http://lewington-pitsos-sache.s3.amazonaws.com/{key}'

print(url)

MB = 1024 * 1024
batch_size = MB * 16
n_iter = 294

def get_chunk(start, end):
    headers = {
        "Range": f"bytes={start}-{end}",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache"
    }

    response = requests.get(url, headers=headers)
    return response

async def async_get_chunk(session, url, start, end):
    headers = {
        "Range": f"bytes={start}-{end}",
    }
    async with session.get(url, headers=headers) as response:
        return start, await response.read()

# async def main():
#     connector = aiohttp.TCPConnector(limit=24)
#     async with aiohttp.ClientSession(connector=connector) as session:
#         start_time = time.time()
        
#         chunks = []
#         for i in range(0, n_iter):
#             start = i * batch_size
#             end = (i + 1) * batch_size - 1
#             chunks.append((start, end))
        
#         tasks = [asyncio.create_task(async_get_chunk(session, url, start, end)) for start, end in chunks]
#         results = await asyncio.gather(*tasks)
    
#         end_time = time.time()
#         elapsed = end_time - start_time
#         print(f"Time taken: {elapsed:.2f} seconds, MB per second: {n_iter * batch_size / MB / elapsed:.2f}")

# asyncio.run(main())

with ProcessPoolExecutor(max_workers=24) as executor:
    start_time = time.time()
    futures = []
    for i in range(0, n_iter):
        
        start = i * batch_size
        end = (i + 1) * batch_size - 1

        futures.append(executor.submit(get_chunk, start, end))

    for future in as_completed(futures):
        chunk = future.result()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Time taken: {elapsed:.2f} seconds, MB per second: {n_iter * batch_size / MB / elapsed:.2f}")
