import threading
import psutil
import json
import time
import torch
from sae_lens import HookedSAETransformer
from torch.utils.data import DataLoader 

from sache.cache import S3WCache, WCache, NoopCache, ThreadedCache
from sache.tok import chunk_and_tokenize
from sache.log import ProcessLogger
from sache.shuffler import ShufflingCache

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

    def log_batch(self, activations,  input_ids):
        self.n += 1
        if self.n % self.log_every == 0:
            sequence_length = activations.shape[1]
            batch_size = activations.shape[0]

            sample_sequence_act = activations[0]
            log_data = {
                'event': 'batch_processed',
                'batches_processed': self.n, 
                
                'activation_shape': activations.shape, 
                'activations_mean': activations.mean().item(), 
                'activations_min': activations.min().item(),
                'activations_max': activations.max().item(),
                'activations_std': activations.std().item(),

                'sample_mean_activations': torch.mean(sample_sequence_act[:, self.sample_activations:], dim=0).tolist(),
                'sample_max_activations': torch.max(sample_sequence_act[:, self.sample_activations:], dim=0).values.tolist(),
                'sample_min_activations': torch.min(sample_sequence_act[:, self.sample_activations:], dim=0).values.tolist(),
                
                'sample_plaintext': self.tokenizer.decode(input_ids[0][:self.sample_activations]),
            }

            if self.start_time is None:
                self.start_time = time.time()
            else:
                current_time = time.time()
                elapsed = current_time - self.start_time
                self.start_time = current_time
                log_data['seconds_since_last_log'] = elapsed
                log_data['samples_per_second'] = self.log_every * batch_size * sequence_length  / elapsed    

            self.log(log_data)

    def finalize(self):
        self._keep_running = False
        self.worker_thread.join()

def build_cache(cache_type, batches_per_cache, run_name, shuffling_buffer_size=16):
    with open('.credentials.json') as f:
        credentials = json.load(f)
    
    if cache_type == 'local':
        cache = WCache(run_name, save_every=batches_per_cache)
        shuffling_cache = ShufflingCache(cache, buffer_size=shuffling_buffer_size)
    elif cache_type == 'local_threaded':
        cache = WCache(run_name, save_every=batches_per_cache)
        shuffling_cache = ThreadedCache(ShufflingCache(cache, buffer_size=shuffling_buffer_size))
    elif cache_type == 's3':
        cache = S3WCache.from_credentials(access_key_id=credentials['AWS_ACCESS_KEY_ID'], secret=credentials['AWS_SECRET'], run_name=run_name, save_every=batches_per_cache)
        shuffling_cache = ShufflingCache(cache, buffer_size=shuffling_buffer_size)
    elif cache_type == 's3_threaded':
        cache = S3WCache.from_credentials(access_key_id=credentials['AWS_ACCESS_KEY_ID'], secret=credentials['AWS_SECRET'], run_name=run_name, save_every=batches_per_cache)
        shuffling_cache = ThreadedCache(ShufflingCache(cache, buffer_size=shuffling_buffer_size))
    elif cache_type == 'noop':
        shuffling_cache = NoopCache()
    else:
        raise ValueError(f"unexpected cache type {cache_type}")

    
    return shuffling_cache

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
    ):

    torch.manual_seed(seed)
    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
    logger = GenerationLogger(run_name, transformer.tokenizer, log_every=log_every)
    logger.log({
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

    cache = build_cache(cache_type, batches_per_cache, run_name)

    dataset = chunk_and_tokenize(
        dataset, 
        transformer.tokenizer, 
        text_key=text_column_name, 
        max_seq_len=max_length,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    transformer.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)

            _, activations = transformer.run_with_cache(
                input_ids, 
                prepend_bos=False, # each sample is actually multiple concatenated samples
                stop_at_layer=layer
            )
            activations = activations[hook_name]

            logger.log_batch(activations, input_ids)


            cache.append(activations.to('cpu'))
        
    cache.finalize()
    logger.finalize()