import boto3
import torch
import json
import time
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.train import SAE, TrainLogger, MeanStdNormalizer
from sache.constants import MB

def main():
    run_name = 'merciless-citadel'
    logger = TrainLogger(run_name, log_mean_std=False)

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    device = 'cuda'
    sae = SAE(n_features=512, hidden_size=768, device=device)
    logger.log_sae(sae)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    normalizer = MeanStdNormalizer('sache/data/normalize/merciless-citadel', device=device)

    cache = S3RCache(s3_client, run_name, 'lewington-pitsos-sache', chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=2)
    
    total_size = cache.metadata['bytes_per_file']
    overall_start = time.time()
    n = 64
    bs = 128
    start = time.time()
    
    for j, t in enumerate(cache):

        for i in range(0, 1024, bs):
            optimizer.zero_grad()
            batch = t[i:i+bs].to(device)
            batch = normalizer.normalize(batch)

            reconstruction, _ = sae(batch)
            rmse = torch.sqrt(torch.mean((batch - reconstruction) ** 2))
            logger.log_loss(rmse, batch)

            if i == 0:
                logger.log_batch(sae, batch, reconstruction)

            rmse.backward()
            optimizer.step()


        end = time.time()
        elapsed = end - start
        logger.log({
            'event': 'file_processed',
            'elapsed': elapsed, 
            'mb_downloaded': total_size / MB, 
            'mbps': total_size / MB / elapsed,
        })

        if j == n - 1:
            break

        start = time.time()


    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n) / (overall_end - overall_start):.2f}")
    cache.stop_downloading()


if __name__ == "__main__":
    main()