import boto3
import torch
import json
import time
import sys 
import os
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache, compile
from sache.train import SAE
from sache.constants import *

def m2():
    mp.set_start_method('spawn')
    buffer=torch.empty((3, 1024, 1024, 768), dtype=torch.float32).share_memory_()
    p = mp.Process(target=compile, args=(buffer, torch.float32, (1024, 1024, 768)))
    p.start()

    p.join()

def main():
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    device = 'cuda'
    sae = SAE(n_features=512, hidden_size=768, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)

    cache = S3RCache(s3_client, 'merciless-citadel', 'lewington-pitsos-sache', chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=2)
    
    total_size = cache.metadata['bytes_per_file']
    overall_start = time.time()
    n = 16
    bs = 128
    
    for j, t in enumerate(cache):
        start = time.time()

        for i in range(0, 1024, bs):
            optimizer.zero_grad()
            batch = t[i:i+bs].to(device)

            reconstruction, _ = sae(batch)
            rmse = torch.sqrt(torch.mean((batch - reconstruction) ** 2))
            print(f"RMSE: {rmse.item()}")
            rmse.backward()
            optimizer.step()

        end = time.time()

        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

        if j == n - 1:
            break

    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n) / (overall_end - overall_start):.2f}")
    cache.stop_downloading()


if __name__ == "__main__":
    main()
