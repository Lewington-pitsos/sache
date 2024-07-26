import time
import torch
from datasets import load_dataset
from sae_lens import HookedSAETransformer
from torch.utils.data import DataLoader 

from sache.cache import Cache
from sache.tok import chunk_and_tokenize
from sache.log import ProcessLogger


def activations_from_batch(transformer, input_ids, layer):
    _, all_hidden_states = transformer.run_with_cache(
        input_ids, 
        prepend_bos=True, 
        stop_at_layer=layer + 1
    )

    return all_hidden_states

class GenerationLogger():
    def __init__(self, cache_dir, tokenizer, log_every=100):
        self.cache_dir = cache_dir
        self.lg = ProcessLogger(cache_dir)
        self.log_every = log_every
        self.n = 0
        self.start_time = time.time()
        self.sample_features = 64
        self.tokenizer = tokenizer

    def log_batch(self, activations, input_ids, attention_mask):
        self.n += 1
        batch_size = activations.shape[0]
        if self.n % self.log_every == 0:
            if self.start_time is None:
                elapsed = 0
            else:
                elapsed = time.time() - self.start_time
            
            sample_sequence_act = activations[0]
            self.lg.log({
                'batches_processed': self.n * batch_size, 
                'seconds_since_last_batch': elapsed, 
                'batches_per_second': self.log_every * batch_size  / elapsed,
                
                'activation_shape': activations.shape, 
                'activations_mean': activations.mean().item(), 
                'activations_min': activations.min().item(),
                'activations_max': activations.max().item(),
                'activations_std': activations.std().item(),

                'sample_mean_activations': torch.mean(sample_sequence_act[:, self.sample_features:], dim=0).tolist(),
                'sample_max_activations': torch.max(sample_sequence_act[:, self.sample_features:], dim=0).tolist(),
                'sample_min_activations': torch.min(sample_sequence_act[:, self.sample_features:], dim=0).tolist(),
                'sample_plaintext': self.tokenizer.decode(input_ids[0][:self.sample_features]),
                'sample_attention_mask': attention_mask[0][:self.sample_features].tolist(),
            })
            self.start_time = time.time()

    def finalize(self):
        pass

def generate(
        cache_dir, 
        dataset, 
        transformer_name, 
        max_length, 
        batch_size, 
        text_column_name, 
        device,
        output_layer,
    ):

    dataset = load_dataset(dataset)
    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
    logger = GenerationLogger(cache_dir, transformer.tokenizer)

    dataset = chunk_and_tokenize(
        dataset, 
        transformer.tokenizer, 
        streaming=True, 
        max_length=max_length, 
        column_name=text_column_name, 
        add_bos_token=True  
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cache = Cache(cache_dir)
    
    transformer.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            activations = activations_from_batch(transformer, input_ids, output_layer)

            logger.log_batch(activations, attention_mask, input_ids)
            attention_mask = attention_mask.unsqueeze(-1).expand_as(activations)
            activations = torch.cat([activations, attention_mask], dim=-1)

            cache.append(activations)

    logger.finalize()
