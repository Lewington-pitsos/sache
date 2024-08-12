# TODO: Implement shuffling

import boto3
import torch
import json
import time
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.train import SAE, TrainLogger, MeanStdNormalizer, NOOPLogger
from sache.constants import MB, BUCKET_NAME

def main():
    n = 32 # 512
    l1_coefficient = 1e-3
    n_feats = 24576
    bs = 256
    samples_per_file = 1024
    inner_bs = 4

    run_name = 'merciless-citadel'

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    # train_logger = TrainLogger(run_name, log_mean_std=True, s3_backup_bucket=BUCKET_NAME, s3_client=s3_client)
    train_logger = NOOPLogger()
    device = 'cuda'
    sae = SAE(n_features=n_feats, hidden_size=768, device=device)

    with train_logger as lg:
        lg.log_sae(sae)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        normalizer = MeanStdNormalizer('sache/normalize/merciless-citadel', device=device)

        cache = S3RCache(s3_client, run_name, 'lewington-pitsos-sache', chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=2)
        
        total_size = cache.metadata['bytes_per_file']
        overall_start = time.time()
        start = time.time()
        
        # no grad
        for j, t in enumerate(cache):
            for i in range(0, samples_per_file, bs):
                st = time.time()
                outer_batch = t[i:i+bs].to(device)
                et = time.time()
                print(f"Time taken to outer batch: {et - st:.2f} seconds")
                for k in range(0, bs, inner_bs):
                    pass
                    optimizer.zero_grad()
                    with torch.no_grad():
                        batch = outer_batch[k:k+inner_bs]
                        # batch = normalizer.normalize(batch)
                        print(batch.is_contiguous(), batch.stride())

                        reconstruction, latent = sae.forward_descriptive(batch)
                    #     mse = ((batch - reconstruction) ** 2).sum(-1).mean()
                    #     l1 = latent.norm(1.0, dim=-1).mean() * l1_coefficient
                    #     loss = mse + l1
                    #     lg.log_loss(mse, l1, loss, batch, latent)

                    #     if i == 0:
                    #         lg.log_batch(sae, batch, reconstruction, latent)

                    #     st = time.time()
                    #     # loss.backward()
                    #     # optimizer.step()
                    #     et = time.time()
                    #     print(f"Time taken to backpropagate: {et - st:.2f} seconds")


            end = time.time()
            elapsed = end - start
            print(f"Time taken for batch {j}: {elapsed:.2f} seconds, MB per second: {total_size / MB / elapsed:.2f}")
            lg.log({
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
