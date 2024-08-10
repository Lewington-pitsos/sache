import os
import torch
from tqdm import tqdm
import numpy as np

from sache.cache import RBatchingCache, RCache, INNER_CACHE_DIR, OUTER_CACHE_DIR
from sache.log import ProcessLogger

class SAE(torch.nn.Module):
    def __init__(self, n_features, hidden_size, device):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.rand(hidden_size, n_features, device=device))
        self.dec = torch.nn.Parameter(torch.rand(n_features, hidden_size, device=device))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        features = self.activation(x @ self.enc)
        return features @ self.dec, features

def build_cache(local_cache_dir, batch_size, device):
    inner_cache = RCache(local_cache_dir, device, buffer_size=8)
    cache = RBatchingCache(cache=inner_cache, batch_size=batch_size)
    return cache

def train(run_name, hidden_size, n_features, device, batch_size=32):
    logger = ProcessLogger(run_name)
    cache_dir = os.path.join(OUTER_CACHE_DIR, run_name, INNER_CACHE_DIR)
    cache = build_cache(cache_dir, batch_size=batch_size, device=device)
    sae = SAE(n_features=n_features, hidden_size=hidden_size, device=device)

    n_batches = 10_000
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for i, activations in tqdm(enumerate(cache), total=n_batches):
        optimizer.zero_grad()

        reconstruction, features = sae(activations)

        rmse = torch.sqrt(torch.mean((activations - reconstruction) ** 2))
        
        rmse.backward()
        optimizer.step()

        logger.log({'event': 'training_batch', 'rmse': rmse.item()})    

        if i > n_batches:
            break

class TrainLogger(ProcessLogger):
    def _get_histogram(self, tensor, bins=50):
        hist = torch.histc(tensor, bins=bins, min=float(tensor.min()), max=float(tensor.max()))

        bin_edges = np.linspace(float(tensor.min()), float(tensor.max()), bins+1)

        hist_list = hist.tolist()
        bin_edges_list = bin_edges.tolist()

        return {
            "counts": hist_list,
            "edges": bin_edges_list
        }

    def log_sae(self, sae):
        self.log({
            'event': 'sae',
            'enc': self._get_histogram(sae.enc),
            'dec': self._get_histogram(sae.dec)
        })
        

if __name__ == '__main__':
    train(
        run_name='active-camera', 
        hidden_size=768, 
        n_features=384, 
        device='cuda', 
        batch_size=256
    )