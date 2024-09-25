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
from sache.model import SwitchSAE, TopKSwitchSAE, TopKSAE, LookupTopkSwitchSAE, SAE
from sache.constants import MB, BUCKET_NAME
from sache.log import NOOPLogger

def save_sae(sae, n_iter, data_name, name, base_dir='cruft'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    model_dir = os.path.join(base_dir, data_name + '-' + name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(sae, os.path.join(model_dir, f'{n_iter}.pt'))

def flatten_activations(t, seq_len, skip_first_n, filter_ma, is_ma, d_in, device):
    if len(t.shape) == 2:
        return t, torch.zeros(t.shape[0], dtype=torch.int64, device=device)

    positions = torch.linspace(0, seq_len - skip_first_n - 1, 0, seq_len - skip_first_n, device=device).repeat(t.shape[0]).to(torch.int64)
    t = t[:, :, :d_in].flatten(0, 1) # (n_samples * (seq_len - 1), d_in)
    if filter_ma:
        flat_ma = is_ma.flatten(0, 1)
        t = t[~flat_ma]
        positions = positions[~flat_ma]

    return t, positions

def main(
        data_name = 'merciless-citadel',
        n_tokens = 32 * 1024 * 1024, # 647 files is the total, 288 means just over 300,000,000 tokens
        k = 32,
        n_feats = 24576,
        d_in = 768,
        batch_size = 4096,
        outer_batch_size = 8192 * 32,
        n_experts = None,
        l1_coefficient = 2e-3,
        privilege_weighting = 1e-2,
        lr = 3e-4,
        tokens_till_latent_dies = 1_000_000,
        device = 'cuda',
        use_wandb=True,
        log_bucket=BUCKET_NAME,
        data_bucket=BUCKET_NAME,
        shuffle=False,
        wandb_project=None,
        name=None, 
        secondary_input=None,
        seq_len=1024,
        skip_first_n=0,
        batch_norm=True,
        filter_ma=False,
        cache_buffer_size=3,
        n_cache_workers=4,
        architecture='topk',
        lr_warmup_steps=None,
        geom_median_file=None,
        save_every=2_500_000,
    ):

    if outer_batch_size < batch_size:
        outer_batch_size = batch_size

    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    train_logger = TrainLogger(data_name, log_mean_std=True, s3_backup_bucket=log_bucket, s3_client=s3_client, use_wandb=use_wandb, wandb_project=wandb_project, log_id=name)
    # train_logger = NOOPLogger()
    if geom_median_file is not None:
        geom_median = torch.load('cruft/geom_median.pt').to(device)
    else:
        geom_median = None

    if n_experts is not None:
        if secondary_input is not None:
            dict = torch.load('cruft/unigrams_gpt2_blocks.10.hook_resid_post_norm.pth', weights_only=True)
            token_lookup = dict[secondary_input]
            sae = LookupTopkSwitchSAE(
                token_lookup=token_lookup, 
                k=k, 
                n_features=n_feats, 
                n_experts=n_experts, 
                d_in=d_in, 
                device=device, 
                efficient=False,
            )
        else:
            sae = TopKSwitchSAE(
                k=k, 
                n_features=n_feats, 
                n_experts=n_experts, 
                d_in=d_in, 
                device=device, 
                efficient=False,
            )
        dead_latents = torch.zeros(n_experts, sae.expert_dim, device=device, requires_grad=False)
    else:
        if architecture == 'topk':
            sae = TopKSAE(k=k, n_features=n_feats, d_in=d_in, device=device, geom_median=geom_median)
        elif architecture == 'relu':
            sae = SAE(n_features=n_feats, d_in=d_in, device=device, geom_median=geom_median)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    with train_logger as lg:
        lg.log_params({
            'k': k,
            'switch_sae': n_experts is not None,
            'skip_first_n': skip_first_n,
            'batch_norm': batch_norm,
            'secondary_input': secondary_input,
            'privilege_weighting': privilege_weighting,
            'n_tokens': n_tokens,
            'n_feats': n_feats,
            'filter_ma': filter_ma,
            'n_experts': n_experts,
            'inner_bs': batch_size,
            'outer_bs': outer_batch_size,
            'learning_rate': lr,
            'l1_coefficient': l1_coefficient,
            'lr_warmup_steps': lr_warmup_steps,
            'architecture': architecture,
            'data_name': data_name,
            'geom_median_file': geom_median_file,
        })
        lg.log_sae(sae)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

        if lr_warmup_steps is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1, epoch / lr_warmup_steps))

        cache = S3RCache(s3_client, data_name, data_bucket, chunk_size=MB * 16, concurrency=200, n_workers=n_cache_workers, buffer_size=cache_buffer_size)

        total_size = cache.metadata['bytes_per_file']
        tokens_per_file = cache.samples_per_file * seq_len

        if shuffle:
            cache = ShufflingRCache(cache, batch_size=outer_batch_size, buffer_size=tokens_per_file * 4, d_in=d_in,  dtype=torch.float32)
        else:
            pass

        overall_start = time.time()
        start = None

        current_files_worth = 0
        token_count = 0
        next_save = save_every
        for acts in cache:
            acts = acts.to(device)  # (n_samples, seq_len, d_in)
            acts = acts[:, skip_first_n:] # (n_samples, seq_len - skip_first_n, d_in)
            is_ma = acts.max(dim=-1).values > 1e3

            acts, positions = flatten_activations(acts, seq_len, skip_first_n, filter_ma, is_ma, d_in, device)

            if secondary_input is not None:
                token_ids = acts[:, :, -1].to(torch.int64).to('cpu').flatten(0, 1)
            else:
                token_ids = None

            for idx in range(0, (acts.shape[0] // batch_size) * batch_size, batch_size):
                token_count += batch_size
                batch = acts[idx:idx+batch_size]
                batch_positions = positions[idx:idx+batch_size]

                optimizer.zero_grad()
                if batch_norm:
                    with torch.no_grad():
                        batch_mean = batch.mean(dim=0, keepdim=True)
                        batch_std = batch.std(dim=0, keepdim=True)
                        batch = (batch - batch_mean) / (batch_std + 1e-6)

                if secondary_input is not None:
                    output = sae.forward_descriptive(batch, token_ids) # (batch_size, d_in), (batch_size, expert_dim), (n_experts, expert_dim)
                else:
                    output = sae.forward_descriptive(batch)
                    
                reconstruction = output['reconstruction']

                if output['active_latents'] is not None:
                    with torch.no_grad():
                        dead_latents[output['active_latents']] = 0
                        dead_latents += batch_size
                        dead_latent_pct = (dead_latents >= tokens_till_latent_dies).to(torch.float32).mean()
                else:
                    dead_latent_pct = None

                delta = batch - reconstruction
                delta_pow = delta.pow(2)
                
                with torch.no_grad():
                    sample_mse = delta_pow.mean(dim=1)
                    if skip_first_n > 0 or filter_ma:
                        mse_sum = torch.bincount(batch_positions, weights=sample_mse)
                        position_counts = torch.bincount(batch_positions)
                        position_mse = mse_sum / torch.clamp(position_counts, min=1)
                    elif seq_len == 1:
                        position_mse = sample_mse.mean().unsqueeze(0)
                    else:
                        position_mse = sample_mse.reshape(-1, seq_len).mean(dim=0)

                    activationwise_variance = batch.pow(2).sum(-1)
                    activationwise_delta = delta_pow.sum(-1)
                    explained_variance = (1 - activationwise_delta / activationwise_variance).mean()


                mse = delta_pow.mean()
                variance_prop_mse = (delta_pow / batch.pow(2).sum(-1, keepdim=True).sqrt()).mean()
                sum_mse = delta_pow.sum(dim=-1).mean()   


                if output['expert_weighting'] is not None:
                    expert_privilege = sae.n_experts * (output['expert_weighting'] * output['expert_prop']).sum()
                    loss = variance_prop_mse + (expert_privilege * privilege_weighting)
                else:
                    expert_privilege = None

                if architecture == 'relu':
                    l1 = output['latent'].abs().sum(dim=1).mean()
                    loss = variance_prop_mse + l1_coefficient * l1
                else:
                    loss = variance_prop_mse
                    l1 = None

                latent = output['latent']
                experts_chosen = output['experts_chosen']

                loss.backward()
                optimizer.step()
                if lr_warmup_steps is not None:
                    scheduler.step()

                lg.log_loss(
                    mse=mse, 
                    sum_mse=sum_mse,
                    l1=l1, 
                    loss=loss, 
                    batch=batch, 
                    latent=latent, 
                    dead_pct=dead_latent_pct, 
                    expert_privilege=expert_privilege,
                    lr=optimizer.param_groups[0]['lr'],
                    position_mse=position_mse,
                    explained_variance=explained_variance,
                    variance_prop_mse=variance_prop_mse,
                    massive_activations=is_ma.to_sparse().indices(),
                )

                

                if token_count >= n_tokens:
                    overall_end = time.time()
                    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
                    return


            files_worth = token_count // tokens_per_file
            if files_worth > current_files_worth:
                current_files_worth = files_worth
                lg.log_batch(sae=sae, batch=batch, reconstruction=reconstruction, latent=latent, experts_chosen=experts_chosen)
                end = time.time()
                if start is not None:
                    elapsed = end - start
                    overall_elapsed = end - overall_start
                    print(f"Time taken for {files_worth} files worth of activations: {elapsed:.2f} seconds, MB per second: {total_size / MB / elapsed:.2f}")
                    lg.log({
                        'event': 'file_processed',
                        'time_to_process_file': elapsed, 
                        'mb_downloaded': total_size / MB, 
                        'mbps': total_size / MB / elapsed,
                        'tokens_per_second': tokens_per_file / elapsed,
                        'total_time_elapsed': overall_elapsed,
                        'file': files_worth,
                    })

                start = time.time()

            if next_save is not None and token_count >= next_save:
                save_sae(sae, token_count, data_name, lg.log_id)
                next_save += save_every
        
        if save_every is not None:
            save_sae(sae, token_count, data_name, lg.log_id)

if __name__ == "__main__":
    fire.Fire(main)
