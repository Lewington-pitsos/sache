import boto3
import torch
import json
import time
import sys 
import os
import multiprocessing as mp
import warnings

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

    cache = S3RCache(s3_client, 'merciless-citadel', 'lewington-pitsos-sache', chunk_size=MB * 8, concurrency=500, n_workers=1)
    
    iter(cache)
    
    total_size = cache.metadata['bytes_per_file']
    for i in range(32):
        start = time.time()

        t = next(cache).to(device)
        print(t.mean())
        print(t.isnan().sum())

        # for i in range(0, 1024, 64):
        #     optimizer.zero_grad()
        #     batch = t[i:i+64]

        #     reconstruction, _ = sae(batch)
        #     rmse = torch.sqrt(torch.mean((batch - reconstruction) ** 2))
        #     print(f"RMSE: {rmse.item()}")
        #     rmse.backward()
        #     optimizer.step()

        end = time.time()


        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

    cache.stop_downloading()


if __name__ == "__main__":
    main()
