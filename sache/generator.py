import threading
import psutil
import json
import os
import time
import torch
from sae_lens import HookedSAETransformer
from torch.utils.data import DataLoader 
from multiprocessing import cpu_count

from sache.cache import S3WCache, WCache, NoopCache, ThreadedWCache
from sache.tok import chunk_and_tokenize
from sache.log import ProcessLogger, NOOPLogger
from sache.shuffler import ShufflingWCache
from sache.constants import BUCKET_NAME
from sache.hookedvit import SpecifiedHookedViT

class GenerationLogger(ProcessLogger):
    def __init__(self, run_name, tokenizer, log_every=100):
        super().__init__(run_name)
        self.tokenizer = tokenizer
        self.log_every = log_every

        self.n = 0
        self.start_time = None
        self.sample_activations = 64

        self.worker_thread = threading.Thread(target=self._log_system_usage)
        self._keep_running = True
        self.worker_thread.start()


    def _log_system_usage(self):
        while self._keep_running:
            self.log({
                'event': 'system_usage',
                'torch.cuda.memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,
                'torch.cuda.memory_reserved': torch.cuda.memory_reserved(0) / 1024**2,
                'torch.cuda.max_memory_reserved': torch.cuda.max_memory_reserved(0) / 1024**2,
                'cpu_percent': psutil.cpu_percent(),
            })
            time.sleep(30)

    def log_batch(self, activations,  input_ids=None):
        self.n += 1
        if self.n % self.log_every == 0:
            batch_size = activations.shape[0]

            if len(activations.shape) == 3:
                sample_sequence_act = activations[0]
            else:
                sample_sequence_act = activations
            
            log_data = {
                'event': 'batch_processed',
                'batches_processed': self.n, 
                
                'activation_shape': activations.shape, 
                'activations_mean': activations.mean().item(), 
                'activations_min': activations.min().item(),
                'activations_max': activations.max().item(),
                'activations_std': activations.std().item(),

                'sample_mean_activations': torch.mean(sample_sequence_act[:, :self.sample_activations], dim=0).tolist(),
                'sample_max_activations': torch.max(sample_sequence_act[:, :self.sample_activations], dim=0).values.tolist(),
                'sample_min_activations': torch.min(sample_sequence_act[:, :self.sample_activations], dim=0).values.tolist(),
            }

            if input_ids is not None:
                assert self.tokenizer is not None
                log_data['sample_plaintext'] = self.tokenizer.decode(input_ids[0][:self.sample_activations])

            if self.start_time is None:
                self.start_time = time.time()
            else:
                current_time = time.time()

                if len(activations.shape) == 3:
                    sequence_length = activations.shape[1]
                    n_tokens = self.log_every * batch_size * sequence_length
                elif len(activations.shape) == 2:
                    n_tokens = self.log_every * batch_size
                else:
                    raise ValueError(f"tried to log unexpected activations shape {activations.shape}")

                elapsed = current_time - self.start_time
                self.start_time = current_time
                log_data['seconds_since_last_log'] = elapsed
                log_data['tokens_per_second'] = n_tokens  / elapsed    

            self.log(log_data)

    def finalize(self):
        self._keep_running = False
        self.worker_thread.join()

def build_cache(cache_type, batches_per_cache, run_name, bucket_name=BUCKET_NAME, shuffling_buffer_size=16):
    with open('.credentials.json') as f:
        cred = json.load(f)
    
    if cache_type == 'local':
        cache = WCache(run_name, save_every=batches_per_cache)
    elif cache_type == 'local_threaded':
        inner_cache = WCache(run_name, save_every=batches_per_cache)
        cache = ThreadedWCache(inner_cache)
    elif cache_type in ['s3r', 's3r_threaded', 's3_threaded_nonshuffling']:
        inner_cache = S3WCache.from_credentials(access_key_id=cred['AWS_ACCESS_KEY_ID'], secret=cred['AWS_SECRET'], run_name=run_name, save_every=batches_per_cache, bucket_name=bucket_name)    
        if cache_type == 's3':
            cache = ShufflingWCache(inner_cache, buffer_size=shuffling_buffer_size)
        elif cache_type == 's3_threaded':
            cache = ThreadedWCache(ShufflingWCache(inner_cache, buffer_size=shuffling_buffer_size))
        elif cache_type == 's3_threaded_nonshuffling':
            cache = ThreadedWCache(inner_cache)
    elif cache_type == 'noop':
        cache = NoopCache()
    else:
        raise ValueError(f"unexpected cache type {cache_type}")

    
    return cache

def generate(
        run_name, 
        batches_per_cache,
        dataset, 
        transformer_name, 
        max_length, 
        batch_size, 
        text_column_name, 
        device,
        layer,
        hook_name,
        cache_type,
        seed=42,
        log_every=100,
        num_proc=cpu_count() // 2,
        bucket_name=None,
    ):

    with open('.credentials.json') as f:
        creds = json.load(f)

    os.environ['HF_TOKEN'] = creds['HF_TOKEN']


    if bucket_name is None:
        bucket_name = BUCKET_NAME

    torch.manual_seed(seed)
    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)

    if log_every is not None:
        logger = GenerationLogger(run_name, transformer.tokenizer, log_every=log_every)
    else:
        logger = NOOPLogger()

    

    with logger as lg:
        lg.log({
            'event': 'start_generating',
            'run_name': run_name,
            'bs_per_cache': batches_per_cache,
            'transformer_name': transformer_name,
            'max_length': max_length,
            'batch_size': batch_size,
            'text_column_name': text_column_name,
            'device': device,
            'layer': layer,
            'hook_name': hook_name,
            'cache_type': cache_type,
            'seed': seed,
        })

        cache = build_cache(cache_type, batches_per_cache, run_name, bucket_name=bucket_name)

        dataset = chunk_and_tokenize(
            dataset, 
            transformer.tokenizer, 
            text_key=text_column_name, 
            max_seq_len=max_length,
            num_proc=num_proc
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        transformer.eval()
        means = None
        stds = None
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)

                _, activations = transformer.run_with_cache(
                    input_ids, 
                    prepend_bos=False, # each sample is actually multiple concatenated samples
                    stop_at_layer=layer
                )
                activations = activations[hook_name]

                if means is None:
                    means = activations.mean(dim=(0, 1)).to('cpu')
                    stds = activations.std(dim=(0, 1)).to('cpu')
                else:
                    means += activations.mean(dim=(0, 1)).to('cpu')
                    stds += activations.std(dim=(0, 1)).to('cpu')

                lg.log_batch(activations, input_ids)

                activations = activations.to('cpu')
                acts_and_ids = torch.concat([activations, input_ids.unsqueeze(2).to('cpu')], dim=2)
                cache.append(acts_and_ids)

            means /= i
            stds /= i

        
        cache.save_mean_std(means, stds)
        cache.finalize()

def vit_generate(
        run_name, 
        batches_per_cache,
        dataset, 
        transformer_name, 
        batch_size, 
        device,
        layer,
        hook_name,
        cache_type,
        n_samples=None,
        seed=42,
        log_every=100,
        bucket_name=None,
    ):

    with open('.credentials.json') as f:
        creds = json.load(f)

    os.environ['HF_TOKEN'] = creds['HF_TOKEN']

    if bucket_name is None:
        bucket_name = BUCKET_NAME

    torch.manual_seed(seed)
    transformer = SpecifiedHookedViT(layer, hook_name, transformer_name, device=device)

    if log_every is not None:
        logger = GenerationLogger(run_name, None, log_every=log_every)
    else:
        logger = NOOPLogger()

    with logger as lg:
        lg.log({
            'event': 'start_generating',
            'run_name': run_name,
            'bs_per_cache': batches_per_cache,
            'transformer_name': transformer_name,
            'batch_size': batch_size,
            'device': device,
            'layer': layer,
            'hook_name': hook_name,
            'cache_type': cache_type,
            'seed': seed,
        })

        cache = build_cache(cache_type, batches_per_cache, run_name, bucket_name=bucket_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        transformer.eval()
        means = None
        stds = None

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                activations = transformer.get_activations(batch)

                if means is None:
                    means = activations.mean(dim=0).to('cpu')
                    stds = activations.std(dim=0).to('cpu')
                else:
                    means += activations.mean(dim=0).to('cpu')
                    stds += activations.std(dim=0).to('cpu')

                lg.log_batch(activations)

                cache.append(activations.to('cpu'))

                if n_samples is not None and i * batch_size >= n_samples:
                    break
                
                if i % 100 == 0:
                    save_mean = means / i
                    save_std = stds / i
                    cache.save_mean_std(save_mean, save_std)

            means /= i
            stds /= i
        
        cache.save_mean_std(means, stds)
        cache.finalize()


46000000