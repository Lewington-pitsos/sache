import json
import time
import torch
from sae_lens import HookedSAETransformer
from torch.utils.data import DataLoader 

from sache.cache import CloudWCache, WCache
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

    def log_batch(self, activations, attention_mask,  input_ids):
        self.n += 1
        if self.n % self.log_every == 0:
            batch_size = activations.shape[0]

            sample_sequence_act = activations[0]
            log_data = {
                'batches_processed': self.n * batch_size, 
                
                'activation_shape': activations.shape, 
                'activations_mean': activations.mean().item(), 
                'activations_min': activations.min().item(),
                'activations_max': activations.max().item(),
                'activations_std': activations.std().item(),

                'sample_mean_activations': torch.mean(sample_sequence_act[:, self.sample_activations:], dim=0).tolist(),
                'sample_max_activations': torch.max(sample_sequence_act[:, self.sample_activations:], dim=0).values.tolist(),
                'sample_min_activations': torch.min(sample_sequence_act[:, self.sample_activations:], dim=0).values.tolist(),
                
                'sample_attention_mask': attention_mask[0][:self.sample_activations].tolist(),
                'attention_mask_shape': attention_mask.shape,
                'attention_mask_sum': attention_mask.sum().item(),
                
                'sample_plaintext': self.tokenizer.decode(input_ids[0][:self.sample_activations]),
            }

            if self.start_time is None:
                self.start_time = time.time()
            else:
                current_time = time.time()
                elapsed = current_time - self.start_time
                self.start_time = current_time
                log_data['seconds_since_last_log'] = elapsed
                log_data['samples_per_second'] = self.log_every * batch_size / elapsed    

            self.log(log_data)

    def finalize(self):
        pass

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
        seed=42,
        cache_source='local'
    ):

    torch.manual_seed(seed)

    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
    with open('.credentials.json') as f:
        credentials = json.load(f)
    
    if cache_source == 'local':
        cache = WCache(run_name, save_every=batches_per_cache)
    elif cache_source == 's3':
        cache = CloudWCache.from_credentials(access_key_id=credentials['AWS_ACCESS_KEY_ID'], secret=credentials['AWS_SECRET'], run_name=run_name, save_every=batches_per_cache)
    else:
        raise ValueError(f"unexpected cache source {cache_source}")

    shuffling_cache = ShufflingCache(cache, buffer_size=8)
    logger = GenerationLogger(run_name, transformer.tokenizer)

    dataset = chunk_and_tokenize(
        dataset, 
        transformer.tokenizer, 
        text_key=text_column_name, 
        max_seq_len=max_length,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    transformer.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = torch.ones_like(input_ids) # attention mask is always 1 BUT we want to be able to save it for cases when it is not (e.g. evals)

            _, activations = transformer.run_with_cache(
                input_ids, 
                prepend_bos=False, # each sample is actually multiple concatenated samples
                stop_at_layer=layer
            )
            activations = activations[hook_name]

            logger.log_batch(activations, attention_mask, input_ids)

            attention_mask = attention_mask.unsqueeze(-1).expand_as(activations)
            activations = torch.cat([activations, attention_mask], dim=-1)

            shuffling_cache.append(activations.to('cpu'))
        
    shuffling_cache.finalize()

    logger.finalize()