import boto3
import torch
import json
import time
import sys 
import os
import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache, ShufflingRCache, RBatchingCache
from sache.train import TrainLogger
from sache.model import SwitchSAE, TopKSwitchSAE, TopKSAE
from sache.constants import MB, BUCKET_NAME
from sache.log import NOOPLogger

# base bs 4096
# base lr 1e-4

def main(
        run_name = 'merciless-citadel',
        n_files = 32, # 647 is the total, 288 means just over 300,000,000 tokens
        k = 32,
        n_feats = 24576,
        d_in = 768,
        batch_size = 4096,
        outer_batch_size = 8192 * 32,
        n_experts = 32,
        l1_coefficient = 2e-3,
        privilege_weighting = 2e-1,
        lr = 3e-4,
        samples_per_file = 1024,
        tokens_till_latent_dies = 10_000_000,
        device = 'cuda',
        use_wandb=True,
        log_bucket=BUCKET_NAME,
        data_bucket=BUCKET_NAME,
        shuffle=True,
        wandb_project=None,
        switch_sae=True,  
        log_id=None, 
        secondary_input=None,
        seq_len=1024,
    ):

    if outer_batch_size < batch_size:
        outer_batch_size = batch_size

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    train_logger = TrainLogger(run_name, log_mean_std=True, s3_backup_bucket=log_bucket, s3_client=s3_client, use_wandb=use_wandb, wandb_project=wandb_project, log_id=log_id)
    # train_logger = NOOPLogger()
    if switch_sae:
        if secondary_input is not None:
            dict = torch.load('cruft/unigrams_gpt2_blocks.10.hook_resid_post_norm.pth', weights_only=True)
            token_lookup = dict[secondary_input]
        else:
            token_lookup = None

        sae = TopKSwitchSAE(
            k=k, 
            n_features=n_feats, 
            n_experts=n_experts, 
            d_in=d_in, 
            device=device, 
            efficient=False, 
            token_lookup=token_lookup,
        )
        dead_latents = torch.zeros(n_experts, sae.latent_dim, device=device, requires_grad=False)
    else:
        sae = TopKSAE(k=k, n_features=n_feats, d_in=d_in, device=device)

    with train_logger as lg:
        lg.log_params({
            'k': k,
            'switch_sae': switch_sae,
            'secondary_input': secondary_input,
            'privilege_weighting': privilege_weighting,
            'n_files': n_files,
            'n_feats': n_feats,
            'n_experts': n_experts,
            'samples_per_file': samples_per_file,
            'inner_bs': batch_size,
            'outer_bs': outer_batch_size,
            'learning_rate': lr,
            'l1_coefficient': l1_coefficient,
        })
        lg.log_sae(sae)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

        cache = S3RCache(s3_client, run_name, data_bucket, chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=3)

        total_size = cache.metadata['bytes_per_file']
        tokens_per_file = samples_per_file * seq_len

        if shuffle:
            cache = ShufflingRCache(cache, batch_size=outer_batch_size, buffer_size=tokens_per_file * 4, d_in=d_in,  dtype=torch.float32)
        else:
            # cache = RBatchingCache(cache, batch_size=outer_batch_size)
            pass

        overall_start = time.time()
        start = None

        token_count = 0
        for t in cache:
            t = t.to(device)
            n_samples = batch_size // seq_len
            for idx in range(0, t.shape[0], n_samples):
                token_count += batch_size

                batch = t[idx:idx+n_samples]

                if secondary_input is not None:
                    token_ids = batch[:, :, -1].to(torch.int64).to('cpu').flatten(0, 1)
                else:
                    token_ids = None

                batch = batch[:, :, :d_in].flatten(0, 1)

                optimizer.zero_grad()
                with torch.no_grad():
                    batch_mean = batch.mean(dim=0, keepdim=True)
                    batch_std = batch.std(dim=0, keepdim=True)
                    batch = (batch - batch_mean) / (batch_std + 1e-6)

                output = sae.forward_descriptive(batch, token_ids) # (batch_size, d_in), (batch_size, expert_dim), (n_experts, expert_dim)
                reconstruction = output['reconstruction']

                if output['active_latents'] is not None:
                    with torch.no_grad():
                        dead_latents[output['active_latents']] = 0
                        dead_latents += batch_size
                        dead_latent_pct = (dead_latents >= tokens_till_latent_dies).to(torch.float32).mean()
                else:
                    dead_latent_pct = None

                mse = ((batch - reconstruction) ** 2).mean()
                mean_pred_mse = ((batch - batch.mean(0)) ** 2).mean()
                scaled_mse = mse / mean_pred_mse    


                if output['expert_weighting'] is not None:
                    expert_privilege = sae.n_experts * (output['expert_weighting'] * output['expert_prop']).sum()
                    loss = scaled_mse + (expert_privilege * privilege_weighting)
                else:
                    expert_privilege = None
                    loss = scaled_mse

                latent = output['latent']
                experts_chosen = output['experts_chosen']

                lg.log_loss(
                    mse=mse, 
                    scaled_mse=scaled_mse,
                    l1=None, 
                    loss=loss, 
                    batch=batch, 
                    latent=latent, 
                    dead_pct=dead_latent_pct, 
                    expert_privilege=expert_privilege,
                    lr=optimizer.param_groups[-1]['lr'],
                    secondary_input=None,
                )

                loss.backward()
                optimizer.step()


            if token_count % tokens_per_file == 0:
                lg.log_batch(sae=sae, batch=batch, reconstruction=reconstruction, latent=latent, experts_chosen=experts_chosen)
                file = token_count // tokens_per_file
                end = time.time()
                if start is not None:
                    elapsed = end - start
                    overall_elapsed = end - overall_start
                    print(f"Time taken for file {file}: {elapsed:.2f} seconds, MB per second: {total_size / MB / elapsed:.2f}")
                    lg.log({
                        'event': 'file_processed',
                        'time_to_process_file': elapsed, 
                        'mb_downloaded': total_size / MB, 
                        'mbps': total_size / MB / elapsed,
                        'total_time_elapsed': overall_elapsed,
                        'file': file,
                    })

                if file >= n_files - 1:
                    break

                start = time.time()

    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n_files) / (overall_end - overall_start):.2f}")

if __name__ == "__main__":
    fire.Fire(main)
