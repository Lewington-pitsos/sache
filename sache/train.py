import os
import torch
from tqdm import tqdm
import numpy as np

from sache.cache import RBatchingCache, RCache, INNER_CACHE_DIR, OUTER_CACHE_DIR
from sache.log import ProcessLogger
from sache.constants import MB  


# implement aux loss to make sure we use all SAE's equally
class SwitchSAE(torch.nn.Module):
    def __init__(self, n_features, n_experts, d_in, device):
        super(SwitchSAE, self).__init__()

        if n_features % n_experts != 0:
            raise ValueError(f'N features {n_features} must be divisible by number of experts {n_experts}')

        self.expert_dim = n_features // n_experts
        self.b_pre = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)

        self.enc = torch.nn.Parameter(
            torch.randn(n_experts, d_in, self.expert_dim, device=device) / (2**0.5) / (d_in ** 0.5)
        )
        self.activation = torch.nn.ReLU()
        self.dec = torch.nn.Parameter(
            torch.randn(n_experts, self.expert_dim, d_in, device=device) / (self.expert_dim) ** 0.5
        )

        self.router_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.router = torch.nn.Parameter(torch.randn(d_in, n_experts, device=device) / (d_in ** 0.5))
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, activations): # activations: (batch_size, d_in)
        recons, _ = self.forward_descriptive(activations)
        return recons

    def forward_descriptive(self, activations): # activations: (batch_size, d_in)
        expert_probabilities = self.softmax((activations - self.router_b) @ self.router) #  (batch_size, n_experts)
        expert_max_prob, expert_idx = torch.max(expert_probabilities, dim=-1) # (batch_size,), (batch_size,)
        
        with torch.no_grad():
            expand_shape = (-1, self.enc.size(1), self.enc.size(2))
            gather_index = expert_idx.view(-1, 1, 1).expand(expand_shape)
            routed_enc = torch.gather(self.enc, 0, gather_index)
            
            gather_index = expert_idx.view(-1, 1, 1).expand(-1, self.dec.size(1), self.dec.size(2))
            routed_dec = torch.gather(self.dec, 0, gather_index)
      
        latent = self.activation(torch.bmm((activations - self.b_pre).unsqueeze(1), routed_enc)) # (batch_size, 1, expert_dim)
        reconstruction = torch.bmm(latent, routed_dec) # (batch_size, 1, d_in)

        reconstruction = expert_max_prob.unsqueeze(1) * reconstruction.squeeze(1) + self.b_pre # (batch_size, d_in)

        return reconstruction, latent.squeeze(1) # (batch_size, d_in), (batch_size, expert_dim)

class SAE(torch.nn.Module):
    def __init__(self, n_features, d_in, device):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.randn(d_in, n_features, device=device) / np.sqrt(n_features))
        self.enc_b = torch.nn.Parameter(torch.randn(n_features, device=device) * 0.01)
        self.dec = torch.nn.Parameter(torch.randn(n_features, d_in, device=device) / np.sqrt(d_in))
        self.dec_b = torch.nn.Parameter(torch.zeros(d_in, dtype=torch.float32, device=device))
        self.activation = torch.nn.ReLU()

    def forward_descriptive(self, x):
        features = self.activation(x @ self.enc + self.enc_b) 
        return features @ self.dec + self.dec_b, features

    def forward(self, x):
        recon, _ = self.forward_descriptive(x)
        return recon
    
def build_cache(local_cache_dir, batch_size, device):
    inner_cache = RCache(local_cache_dir, device, buffer_size=8)
    cache = RBatchingCache(cache=inner_cache, batch_size=batch_size)
    return cache

def train(run_name, d_in, n_features, device, batch_size=32):
    logger = ProcessLogger(run_name)
    cache_dir = os.path.join(OUTER_CACHE_DIR, run_name, INNER_CACHE_DIR)
    cache = build_cache(cache_dir, batch_size=batch_size, device=device)
    sae = SAE(n_features=n_features, d_in=d_in, device=device)

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
        with torch.no_grad():
            return (x - self.mean) / self.std

class TrainLogger(ProcessLogger):
    def __init__(self, run_name, log_mean_std=False, *args, **kwargs):
        super(TrainLogger, self).__init__(run_name, *args, **kwargs)
        self.log_mean_std = log_mean_std

    def log_sae(self, sae, info=None):
        with torch.no_grad():
            ecounts, eedges = get_histogram(sae.enc)
            ebcounts, ebedges = get_histogram(sae.enc_b, bins=25)
            dcounts, dedges = get_histogram(sae.dec)
            dbcounts, dbedges = get_histogram(sae.dec_b, bins=25)
            
        message = {
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
            'dec_b': {
                'counts': dbcounts,
                'edges': dbedges
            }
        }

        if info is not None:
            for k in info.keys():
                if k in message:
                    raise ValueError(f'Key {k} already exists in message', message, info)
            message.update(info)
        
        self.log(message)

    def log_loss(self, mse, l1, loss, batch, latent):
        with torch.no_grad():
            message = {
                'event': 'training_batch', 
                'mse': mse.item(),
                'L0': (latent > 0).float().sum(-1).mean().item(),
                'L1': l1.item(),
                'loss': loss.item()
            }
                    
            if self.log_mean_std:
                message.update({
                    'input_mean': batch.mean(dim=(0, 1)).cpu().numpy().tolist(), 
                    'input_std': batch.std(dim=(0, 1)).cpu().numpy().tolist()
                })

            self.log(message)

    def log_batch(self, sae, batch, reconstruction, latent):
        with torch.no_grad():
            binput, einput = get_histogram(batch)
            brecon, erecon = get_histogram(reconstruction)
            bdelta, edelta = get_histogram(batch - reconstruction)
            blatent, elatent = get_histogram(latent)

        info = {
            'input_hist': { 'counts': binput, 'edges': einput},
            'reconstruction_hist': { 'counts': brecon, 'edges': erecon},
            'delta_hist': { 'counts': bdelta, 'edges': edelta},
            'latent_hist': { 'counts': blatent, 'edges': elatent},
        }

        self.log_sae(sae, info=info)

class NOOPLogger:
    def log(self, data):
        # This method can handle logging data, currently it does nothing.
        pass

    def __enter__(self):
        # Allows the logger to be used as a context manager.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Handle exiting the context manager, ignoring any exceptions.
        pass

    def __getattr__(self, name):
        # This will catch any undefined attribute or method calls.
        # It returns a lambda that does nothing, making every call a no-op.
        return lambda *args, **kwargs: None


if __name__ == '__main__':
    train(
        run_name='active-camera', 
        d_in=768, 
        n_features=384, 
        device='cuda', 
        batch_size=256
    )