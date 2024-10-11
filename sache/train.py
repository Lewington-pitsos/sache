import os
import time

import torch
import numpy as np
import boto3

from sache.log import SacheLogger, LOG_DIR
from sache.model import SAE, LookupTopkSwitchSAE, SwitchSAE, TopKSwitchSAE, TopKSAE
from sache.cache import S3RCache, ShufflingRCache
from sache.constants import MB

def get_histogram(tensor, bins=50):
    tensor = tensor.detach()
    hist = torch.histc(tensor, bins=bins, min=float(tensor.min()), max=float(tensor.max()))

    bin_edges = np.linspace(float(tensor.min()), float(tensor.max()), bins+1)

    hist_list = hist.tolist()
    bin_edges_list = bin_edges.tolist()

    return hist_list, bin_edges_list

class TrainLogger(SacheLogger):
    def __init__(self, run_name, log_mean_std=False, max_sample=1024, *args, **kwargs):
        super(TrainLogger, self).__init__(run_name, *args, **kwargs)
        self.log_mean_std = log_mean_std
        self.max_sample = max_sample

    def log_sae(self, sae, info=None):
        if isinstance(sae, SAE):
            message = self._log_sae(sae)
        elif isinstance(sae, TopKSAE):
            message = self._log_sae(sae)
        elif isinstance(sae, TopKSwitchSAE):
            message = self._log_switch_sae(sae)
        elif isinstance(sae, SwitchSAE):
            message = self._log_switch_sae(sae)
        else:
            raise ValueError(f'Unknown SAE type {type(sae)}')

        if info is not None:
            for k in info.keys():
                if k in message:
                    raise ValueError(f'Key {k} already exists in message', message, info)
            message.update(info)
        
        self.log(message)

    def _log_switch_sae(self, sae, info=None):
        with torch.no_grad():
            ecounts, eedges = get_histogram(sae.enc)
            dcounts, dedges = get_histogram(sae.dec)
            routercounts, routeredges = get_histogram(sae.router)
            broutercounts, brouteredges = get_histogram(sae.router_b, bins=25)
            bprecounts, bpreedges = get_histogram(sae.pre_b, bins=25)
            
        return {
            'event': 'sae',
            'enc_experts': { 
                'counts': ecounts,
                'edges': eedges
            },
            'pre_b': {
                'counts': bprecounts,
                'edges': bpreedges
            },
            'dec_experts': {
                'counts': dcounts,
                'edges': dedges
            },
            'router_b': {
                'counts': broutercounts,
                'edges': brouteredges
            },
            'router': {
                'counts': routercounts,
                'edges': routeredges
            }
        }

    def _log_sae(self, sae, info=None):
        with torch.no_grad():
            ecounts, eedges = get_histogram(sae.enc)
            ebcounts, ebedges = get_histogram(sae.pre_b, bins=25)
            dcounts, dedges = get_histogram(sae.dec)
            
        return {
            'event': 'sae',
            'enc': { 
                'counts': ecounts,
                'edges': eedges
            },
            'enc_b': {
                'counts': ebcounts,
                'edges': ebedges
            },
            'dec': {
                'counts': dcounts,
                'edges': dedges
            },
        }

    def log_loss(self, mse, sum_mse, l1, loss, batch, latent, dead_pct, 
                expert_privilege, lr, position_mse, explained_variance, 
                variance_prop_mse):
        with torch.no_grad():
            message = {
                'event': 'training_batch', 
                'mse': mse.item(),
                'sum_mse': sum_mse.item(),
                'loss': loss.item(),
                'batch_learning_rate': lr
            }

            if variance_prop_mse is not None:
                message['variance_proportional_mse'] = variance_prop_mse.item()

            if explained_variance is not None:
                message['explained_variance'] = explained_variance.item()

            if position_mse is not None:
                message['position_mse'] = position_mse.cpu().numpy().tolist()

            if latent is not None:
                message['L0'] = (latent > 0).float().sum(-1).mean().item()

            if dead_pct is not None:
                message['dead_feature_prop'] = dead_pct.item()

            if expert_privilege is not None:
                message['expert_privilege'] = expert_privilege.item()

            if l1 is not None:
                message['l1'] = l1.item()

            if self.log_mean_std:
                message.update({
                    'input_mean': batch.mean(dim=(0, 1)).cpu().numpy().tolist(), 
                    'input_std': batch.std(dim=(0, 1)).cpu().numpy().tolist()
                })

            self.log(message)

    def log_batch(self, sae, batch, reconstruction, latent, experts_chosen):
        batch = batch[:self.max_sample]
        reconstruction = reconstruction[:self.max_sample]
        latent = latent[:self.max_sample]

        with torch.no_grad():
            binput, einput = get_histogram(batch)
            brecon, erecon = get_histogram(reconstruction)
            bdelta, edelta = get_histogram(batch - reconstruction)
            blatent, elatent = get_histogram(latent)

            bencgrad, eencgrad = get_histogram(sae.enc.grad)
            bdecgrad, edecgrad = get_histogram(sae.dec.grad)
            bpregrad, epregrad = get_histogram(sae.pre_b.grad)
            bdec, edec = get_histogram(sae.dec)


            info = {
                'input_hist': { 'counts': binput, 'edges': einput},
                'reconstruction_hist': { 'counts': brecon, 'edges': erecon},
                'delta_hist': { 'counts': bdelta, 'edges': edelta},
                'latent_hist': { 'counts': blatent, 'edges': elatent},
                
                'dec_hist': { 'counts': bdec, 'edges': edec},

                'enc_grad_hist': { 'counts': bencgrad, 'edges': eencgrad},
                'dec_grad_hist': { 'counts': bdecgrad, 'edges': edecgrad},
                'pre_grad_hist': { 'counts': bpregrad, 'edges': epregrad}
            }

            if experts_chosen is not None:
                experts_chosen = experts_chosen[:self.max_sample]
                bexperts, eexperts = get_histogram(experts_chosen, bins=sae.n_experts)

                info['experts_chosen_hist'] = { 'counts': bexperts, 'edges': eexperts}

            if hasattr(sae, 'k'):
                info['k'] = sae.k

            if hasattr(sae, 'router'):
                broutergrad, eroutergrad = get_histogram(sae.router.grad)
                info['router_grad_hist'] = { 'counts': broutergrad, 'edges': eroutergrad}

        self.log_sae(sae, info=info)

def save_sae(sae, n_iter, data_name, name, base_dir='log', s3_client=None, bucket_name=None):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    model_dir = os.path.join(base_dir, data_name, name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = os.path.join(model_dir, f'{n_iter}.pt')
    torch.save(sae, model_filename)

    if s3_client is not None:
        
        if bucket_name is None:
            raise ValueError('bucket_name must be provided if s3_client is provided')
        s3_path = f'{base_dir}/{data_name}/{name}/{n_iter}.pt'

        print(f'Uploading {model_filename} to {bucket_name}/{s3_path}')
        s3_client.upload_file(model_filename, bucket_name, s3_path)

def flatten_activations(t, seq_len, skip_first_n, d_in, device):
    if len(t.shape) == 2:
        return t, torch.zeros(t.shape[0], dtype=torch.int64, device=device)

    positions = torch.linspace(0, seq_len - skip_first_n - 1, seq_len - skip_first_n, device=device).repeat(t.shape[0]).to(torch.int64)
    t = t[:, :, :d_in].flatten(0, 1)
    return t, positions

def load_sae_from_checkpoint(checkpoint, s3_client, local_dir='cruft'):
    if checkpoint.startswith('s3://'):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        local_path = f'{local_dir}/{os.path.basename(checkpoint)}'
        s3_client.download_file(
            Bucket=checkpoint.split('/')[2], 
            Key='/'.join(checkpoint.split('/')[3:]), 
            Filename=local_path
        )
        return torch.load(local_path)
    else:
        return torch.load(checkpoint)

def train_sae(
        data_name,
        credentials,
        log_bucket,
        data_bucket,
        n_tokens = 32 * 1024 * 1024, # 647 files is the total, 288 means just over 300,000,000 tokens
        k = 32,
        n_feats = 24576,
        d_in = 768,
        batch_size = 4096,
        n_experts = None,
        l1_coefficient = 2e-3,
        privilege_weighting = 1e-2,
        lr = 3e-4,
        tokens_till_latent_dies = 10_000_000,
        device = 'cuda',
        use_wandb=True,
        shuffle=False,
        wandb_project=None,
        name=None, 
        secondary_input=None,
        seq_len=1024,
        skip_first_n=0,
        batch_norm=True,
        cache_buffer_size=3,
        n_cache_workers=4,
        architecture='topk',
        lr_warmup_steps=None,
        geom_median_file=None,
        save_every=2_500_000,
        save_checkpoints_to_s3=False,
        load_checkpoint=None,
        start_from=None,
        base_log_dir=LOG_DIR
    ):
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    train_logger = TrainLogger(
        data_name, 
        base_dir=base_log_dir,
        log_mean_std=True, 
        s3_backup_bucket=log_bucket, 
        s3_client=s3_client, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        log_id=name,
        credentials=credentials,
    )
    
    if geom_median_file is not None:
        geom_median = torch.load('cruft/geom_median.pt').to(device)
    else:
        geom_median = None

    if load_checkpoint is not None:
        sae = load_sae_from_checkpoint(load_checkpoint, s3_client)
    elif n_experts is not None:
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
            'n_experts': n_experts,
            'batch_size': batch_size,
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

        cache = S3RCache(s3_client, data_name, data_bucket, chunk_size=MB * 16, concurrency=200, n_workers=n_cache_workers, buffer_size=cache_buffer_size, start_from=start_from)
        print('total number of files to download', len(cache))

        total_size = cache.metadata['bytes_per_file']
        tokens_per_file = cache.samples_per_file * seq_len

        if shuffle:
            cache = ShufflingRCache(cache, batch_size=tokens_per_file, buffer_size=tokens_per_file * 4, d_in=d_in,  dtype=torch.float32)
        else:
            pass

        overall_start = time.time()
        start = None

        current_files_worth = 0
        token_count = 0
        next_save = save_every
        with cache as running_cache:
            for acts in running_cache:
                acts = acts.to(device)  # (n_samples, seq_len, d_in)
                acts = acts[:, skip_first_n:] # (n_samples, seq_len - skip_first_n, d_in)

                acts, positions = flatten_activations(acts, seq_len, skip_first_n, d_in, device)

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
                        if skip_first_n > 0:
                            mse_sum = torch.bincount(batch_positions, weights=sample_mse)
                            position_counts = torch.bincount(batch_positions)
                            position_mse = mse_sum / torch.clamp(position_counts, min=1)
                        elif seq_len == 1:
                            position_mse = sample_mse.mean().unsqueeze(0)
                        else:
                            position_mse = sample_mse.reshape(-1, seq_len).mean(dim=0)

                        batch_mean = batch.mean(-1, keepdim=True)
                        delta_mean = delta.mean(-1, keepdim=True)

                        activation_variance = batch_mean.pow(2).sum(-1)
                        delta_variance = delta_mean.pow(2).sum(-1)
                        explained_variance = (1 - delta_variance / activation_variance).mean()


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
                    )

                    if token_count >= n_tokens:
                        overall_end = time.time()
                        print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
                        break

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
                    save_sae(
                        sae, 
                        token_count, 
                        data_name, 
                        lg.log_id, 
                        s3_client=s3_client if save_checkpoints_to_s3 else None, 
                        bucket_name=log_bucket
                    )
                    next_save += save_every

                if token_count >= n_tokens:
                    break

        # out of the cache download loop
        if save_every is not None:
            save_sae(
                sae, 
                token_count, 
                data_name, 
                lg.log_id, 
                s3_client=s3_client if save_checkpoints_to_s3 else None, 
                bucket_name=log_bucket
            )

