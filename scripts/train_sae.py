import boto3
import torch
import json
import time
import sys 
import os
import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache, ShufflingRCache, RBatchingCache
from sache.train import TrainLogger, SwitchSAE, TopKSwitchSAE
from sache.constants import MB, BUCKET_NAME

def main(
        run_name = 'merciless-citadel',
        n_steps = 289, # 647 is the total, 288 means just over 300,000,000 tokens
        k = 32,
        n_feats = 24576,
        d_in = 768,
        batch_size = 8192 * 32,
        n_experts = 32,
        l1_coefficient = 2e-3,
        privilege_weighting = 2e-1,
        learning_rate = 1e-3,
        samples_per_file = 1024,
        tokens_till_latent_dies = 10_000_000,
        device = 'cuda',
        use_wandb=True,
        log_bucket=BUCKET_NAME,
        data_bucket=BUCKET_NAME,
        shuffle=True,
        wandb_project=None,
        base_expert=False,
    ):
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    train_logger = TrainLogger(run_name, log_mean_std=True, s3_backup_bucket=log_bucket, s3_client=s3_client, use_wandb=use_wandb, wandb_project=wandb_project)
    sae = TopKSwitchSAE(
        k=k, 
        n_features=n_feats, 
        n_experts=n_experts, 
        d_in=d_in, 
        device=device, 
        efficient=False, 
        base_expert=base_expert,
    )

    dead_latents = torch.zeros(n_experts, sae.latent_dim, device=device, requires_grad=False)

    with train_logger as lg:
        lg.log_params({
            'k': k,
            'privilege_weighting': privilege_weighting,
            'n_steps': n_steps,
            'n_feats': n_feats,
            'n_experts': n_experts,
            'samples_per_file': samples_per_file,
            'inner_bs': batch_size,
            'learning_rate': learning_rate,
            'l1_coefficient': l1_coefficient,
        })
        lg.log_sae(sae)
        optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

        cache = S3RCache(s3_client, run_name, data_bucket, chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=3)

        dataset_mean = torch.tensor(cache.metadata['mean'], device=device, requires_grad=False)
        dataset_std = torch.tensor(cache.metadata['std'], device=device, requires_grad=False)

        total_size = cache.metadata['bytes_per_file']
        tokens_per_file = samples_per_file * 1024

        if shuffle:
            cache = ShufflingRCache(cache, batch_size=batch_size, buffer_size=tokens_per_file * 2, d_in=d_in,  dtype=torch.float32)
        else:
            cache = RBatchingCache(cache, batch_size=batch_size)

        overall_start = time.time()
        start = time.time()
        
        token_count = 0
        for j, t in enumerate(cache):
            token_count += batch_size

            optimizer.zero_grad()
            batch = t.to(device)
            with torch.no_grad():
                batch = (batch - dataset_mean) / dataset_std

            output = sae.forward_descriptive(batch) # (batch_size, d_in), (batch_size, expert_dim), (n_experts, expert_dim)
            reconstruction = output['reconstruction']

            with torch.no_grad():
                dead_latents[output['active_latents']] = 0
                dead_latents += batch_size
                dead_latent_pct = (dead_latents >= tokens_till_latent_dies).to(torch.float32).mean()
            
            mse = ((batch - reconstruction) ** 2).sum(-1).mean()
            mean_pred_mse = ((batch - batch.mean(0)) ** 2).sum(-1).mean()
            scaled_mse = mse / mean_pred_mse

            expert_privilege = sae.n_experts * (output['expert_weighting'] * output['expert_prop']).sum()

            loss = scaled_mse + (expert_privilege * privilege_weighting)
            lg.log_loss(
                mse=mse, 
                scaled_mse=scaled_mse, 
                l1=None, 
                loss=loss, 
                batch=batch, 
                latent=output['latent'], 
                dead_pct=dead_latent_pct, 
                expert_privilege=expert_privilege,
                lr=optimizer.param_groups[-1]['lr'],
            )

            loss.backward()
            optimizer.step()

            if token_count % tokens_per_file == 0:
                lg.log_batch(sae=sae, batch=batch, reconstruction=reconstruction, latent=output['latent'], experts_chosen=output['experts_chosen'])
                end = time.time()
                elapsed = end - start
                print(f"Time taken for file {j}: {elapsed:.2f} seconds, MB per second: {total_size / MB / elapsed:.2f}")
                lg.log({
                    'event': 'file_processed',
                    'elapsed': elapsed, 
                    'mb_downloaded': total_size / MB, 
                    'mbps': total_size / MB / elapsed,
                })

                if token_count // tokens_per_file == n_steps - 1:
                    break

                start = time.time()

    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n_steps) / (overall_end - overall_start):.2f}")

if __name__ == "__main__":
    fire.Fire(main)
