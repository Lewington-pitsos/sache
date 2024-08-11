import os
import torch
from tqdm import tqdm
import numpy as np

from sache.cache import RBatchingCache, RCache, INNER_CACHE_DIR, OUTER_CACHE_DIR
from sache.log import ProcessLogger
from sache.constants import MB  

class SAE(torch.nn.Module):
    def __init__(self, n_features, hidden_size, device):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.randn(hidden_size, n_features, device=device) / np.sqrt(n_features))
        self.dec = torch.nn.Parameter(torch.randn(n_features, hidden_size, device=device) / np.sqrt(hidden_size))
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

def get_histogram(tensor, bins=50):
    hist = torch.histc(tensor, bins=bins, min=float(tensor.min()), max=float(tensor.max()))

    bin_edges = np.linspace(float(tensor.min()), float(tensor.max()), bins+1)

    hist_list = hist.tolist()
    bin_edges_list = bin_edges.tolist()

    return hist_list, bin_edges_list

class MeanStdNormalizer():
    def __init__(self, parent_dir, device):
        self.mean = torch.load(os.path.join(parent_dir, 'mean.pt'), map_location=device, weights_only=True)
        self.std = torch.load(os.path.join(parent_dir, 'std.pt'), map_location=device, weights_only=True)

    def normalize(self, x):
        return (x - self.mean) / self.std

class TrainLogger(ProcessLogger):
    def __init__(self, run_name, log_mean_std=False):
        super(TrainLogger, self).__init__(run_name)
        self.log_mean_std = log_mean_std

    def log_sae(self, sae, info=None):
        
        ecounts, eedges = get_histogram(sae.enc)
        dcounts, dedges = get_histogram(sae.dec)
        message = {
            'event': 'sae',
            'enc': { 
                'counts': ecounts,
                'edges': eedges
            },
            'dec': {
                'counts': dcounts,
                'edges': dedges
            }
        }

        if info is not None:
            for k in info.keys():
                if k in message:
                    raise ValueError(f'Key {k} already exists in message', message, info)
            message.update(info)
        
        self.log(message)

    def log_loss(self, rmse, batch):
        message = {'event': 'training_batch', 'rmse': rmse.item()}
        
        if self.log_mean_std:
            message.update({
                'input_mean': batch.mean(dim=(0, 1)).cpu().numpy().tolist(), 
                'input_std': batch.std(dim=(0, 1)).cpu().numpy().tolist()
            })

        self.log(message)

    def log_batch(self, sae, batch, reconstruction):
        binput, einput = get_histogram(batch)
        brecon, erecon = get_histogram(reconstruction)
        bdelta, edelta = get_histogram(batch - reconstruction)
        info = {
            'input_hist': { 'counts': binput, 'edges': einput},
            'reconstruction_hist': { 'counts': brecon, 'edges': erecon},
            'delta_hist': { 'counts': bdelta, 'edges': edelta}
        }

        self.log_sae(sae, info=info)

if __name__ == '__main__':
    train(
        run_name='active-camera', 
        hidden_size=768, 
        n_features=384, 
        device='cuda', 
        batch_size=256
    )