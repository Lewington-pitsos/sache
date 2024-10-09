import threading
import psutil
import os
import time
import torch
from torch.utils.data import DataLoader 

from sache.cache import S3WCache, WCache, NoopCache, ThreadedWCache, MultiLayerS3WCache
from sache.log import SacheLogger, NOOPLogger
from sache.shuffler import ShufflingWCache
from sache.hookedvit import SpecifiedHookedViT

class GenerationLogger(SacheLogger):
    def __init__(self, run_name, tokenizer, *args, log_every=100,  **kwargs):
        super().__init__(run_name, *args, **kwargs)
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
                'cpu_memory_percent': psutil.virtual_memory().percent,
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

def build_cache(creds, cache_type, batches_per_cache, run_name, bucket_name, shuffling_buffer_size=16, **kwargs):
    if cache_type == 'local':
        cache = WCache(run_name, save_every=batches_per_cache)
    elif cache_type == 's3_multilayer':
        cache = MultiLayerS3WCache(
            creds=creds,
            run_name=run_name,
            max_queue_size=batches_per_cache, 
            bucket_name=bucket_name,
            **kwargs
        )
    elif cache_type == 'local_threaded':
        inner_cache = WCache(run_name, save_every=batches_per_cache)
        cache = ThreadedWCache(inner_cache)
    elif cache_type in ['s3', 's3_nonshuffling', 's3_threaded', 's3_threaded_nonshuffling']:
        inner_cache = S3WCache.from_credentials(
            access_key_id=creds['AWS_ACCESS_KEY_ID'], 
            secret=creds['AWS_SECRET'], 
            run_name=run_name, 
            save_every=batches_per_cache, 
            bucket_name=bucket_name
        )    
        if cache_type == 's3':
            cache = ShufflingWCache(inner_cache, buffer_size=shuffling_buffer_size)
        if cache_type == 's3_nonshuffling':
            cache = inner_cache
        elif cache_type == 's3_threaded':
            cache = ThreadedWCache(ShufflingWCache(inner_cache, buffer_size=shuffling_buffer_size))
        elif cache_type == 's3_threaded_nonshuffling':
            cache = ThreadedWCache(inner_cache)
    elif cache_type == 'noop':
        cache = NoopCache()
    else:
        raise ValueError(f"unexpected cache type {cache_type}")

    
    return cache

def vit_generate(
        creds,
        run_name, 
        batches_per_cache,
        dataset, 
        transformer_name, 
        batch_size, 
        device,
        hook_locations,
        cache_type,
        bucket_name,
        n_samples=None,
        seed=42,
        log_every=100,
        full_sequence=False,
        num_data_workers=2,
        input_tensor_shape=None,
        num_cache_workers=5,
        print_logs=False,
    ):

    torch.manual_seed(seed)
    os.environ['HF_TOKEN'] = creds['HF_TOKEN']
    if log_every is not None:
        logger = GenerationLogger(run_name, None, log_every=log_every, print_logs=print_logs)
    else:
        logger = NOOPLogger()

    transformer = SpecifiedHookedViT(hook_locations, transformer_name, device=device)
    with logger as lg:
        lg.log({
            'event': 'start_generating',
            'run_name': run_name,
            'bs_per_cache': batches_per_cache,
            'transformer_name': transformer_name,
            'batch_size': batch_size,
            'device': device,
            'cache_type': cache_type,
            'seed': seed,
            'num_data_workers': num_data_workers,
            'num_cache_workers': num_cache_workers,
            'hook_locations': hook_locations,
        })

        cache = build_cache(
            creds=creds,
            cache_type=cache_type, 
            batches_per_cache=batches_per_cache, 
            run_name=run_name, 
            bucket_name=bucket_name,

            num_workers=num_cache_workers,
            input_tensor_shape=input_tensor_shape,
            hook_locations=hook_locations, 
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers)
        
        transformer.eval()

        with cache as c:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    
                    if full_sequence:
                        cache_dict = transformer.all_activations(batch)
                    else:
                        cache_dict = transformer.cls_activations(batch)

                    c.append(cache_dict)

                    if n_samples is not None and i * batch_size >= n_samples:
                        break

                    lg.log_batch(cache_dict[hook_locations[-1]].to('cpu')) # only logs the last activations
