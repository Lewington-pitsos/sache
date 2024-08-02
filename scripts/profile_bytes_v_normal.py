import requests
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
import boto3
import torch

full_start = time.time()

def generate_random_tensor(size_mb):
    tensor_size = round((size_mb * 1024**2)) // 4  # float32 has 4 bytes
    tensor = torch.randn(tensor_size // (1024), 1024)
    return tensor

with open('.credentials.json') as f:
    credentials = json.load(f)
s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

tensor = generate_random_tensor(1 * 200)

buffer = io.BytesIO()
torch.save(tensor, buffer)
buffer.seek(0)

key = 'tensors/profile.pt'
s3_client.put_object(
    Bucket='lewington-pitsos-sache', 
    Key=key, 
    Body=buffer, 
    ContentType='application/octet-stream'
)

elapsed = time.time() - full_start
print(f'finished uploading, seconds elapsed {elapsed:.2f}')

url = f'http://lewington-pitsos-sache.s3.amazonaws.com/{key}'

MB = 1024 * 1024
batch_size = MB * 1
n_iter = 100

def get_chunk(start, end):
    headers = {
        "Range": f"bytes={start}-{end}",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache"
    }

    response = requests.get(url, headers=headers)
    return 0

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