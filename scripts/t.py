# TODO: Implement shuffling

import boto3
import torch
import json
import time
import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.train import SAE, TrainLogger, MeanStdNormalizer, NOOPLogger, SwitchSAE, TopKSwitchSAE
from sache.constants import MB, BUCKET_NAME

def main():
    n_steps = 700 # 647 is the total
    k = 32
    l1_coefficient = 1e-3
    n_feats = 24576
    d_in = 768
    batch_size = 8192 * 32 
    n_experts = 32
    learning_rate = 1e-4
    samples_per_file = 1024
    tokens_till_latent_dies = 10_000_000
    device = 'cuda'

    run_name = 'merciless-citadel'

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    train_logger = TrainLogger(run_name, log_mean_std=True, s3_backup_bucket=BUCKET_NAME, s3_client=s3_client)
    # train_logger = NOOPLogger()
    sae = TopKSwitchSAE(k=k, n_features=n_feats, n_experts=n_experts, d_in=d_in, device=device, efficient=False)

    dead_latents = torch.zeros(n_experts, n_feats // n_experts, device=device)

    with train_logger as lg:
        lg.log({
            'k': k,
            'n_steps': n_steps,
            'l1_coefficient': l1_coefficient,
            'n_feats': n_feats,
            'n_experts': n_experts,
            'samples_per_file': samples_per_file,
            'inner_bs': batch_size,
            'learning_rate': learning_rate,
        })
        lg.log_sae(sae)
        optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
        normalizer = MeanStdNormalizer('sache/normalize/merciless-citadel', device=device)

        cache = S3RCache(s3_client, run_name, BUCKET_NAME, chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=2)
        
        total_size = cache.metadata['bytes_per_file']
        overall_start = time.time()
        start = time.time()
        
        for j, t in enumerate(cache):
            t = t.flatten(0, 1) # t comes out as (batch_size, sequence_length, d_in), we want to pretend there is no sequence, each sample is a token now.
            for k in range(0, t.shape[0], batch_size):
                optimizer.zero_grad()
                batch = t[k:k+batch_size].to(device)
                batch = normalizer.normalize(batch)

                reconstruction, latent, was_active = sae.forward_descriptive(batch) # (batch_size, d_in), (batch_size, expert_dim), (n_experts, expert_dim)
                
                with torch.no_grad():
                    dead_latents[was_active] = 0
                    dead_latents += batch_size

                dead_latent_pct = (dead_latents >= tokens_till_latent_dies).sum() / dead_latents.numel()
                
                mse = ((batch - reconstruction) ** 2).sum(0).mean()
                mean_pred_mse = ((batch - batch.mean(0)) ** 2).sum(0).mean()
                scaled_mse = mse / mean_pred_mse
                
                loss = scaled_mse
                lg.log_loss(mse=mse, scaled_mse=scaled_mse, l1=None, loss=loss, batch=batch, latent=latent, dead_pct=dead_latent_pct)

                if k == 0:
                    lg.log_batch(sae, batch, reconstruction, latent)

                loss.backward()
                optimizer.step()


            end = time.time()
            elapsed = end - start
            print(f"Time taken for batch {j}: {elapsed:.2f} seconds, MB per second: {total_size / MB / elapsed:.2f}")
            lg.log({
                'event': 'file_processed',
                'elapsed': elapsed, 
                'mb_downloaded': total_size / MB, 
                'mbps': total_size / MB / elapsed,
            })

            if j == n_steps - 1:
                break

            start = time.time()


    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n_steps) / (overall_end - overall_start):.2f}")
    cache.stop_downloading()


if __name__ == "__main__":
    main()
