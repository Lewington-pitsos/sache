import os
import torch
from tqdm import tqdm
import numpy as np

from sache.cache import RBatchingCache, RCache, INNER_CACHE_DIR, OUTER_CACHE_DIR
from sache.log import ProcessLogger
from sache.constants import MB  
from sache.model import SAE, SwitchSAE, TopKSwitchSAE

def build_cache(local_cache_dir, batch_size, device):
    inner_cache = RCache(local_cache_dir, device, buffer_size=8)
    cache = RBatchingCache(cache=inner_cache, batch_size=batch_size)
    return cache

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
        if isinstance(sae, SAE):
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
            ebcounts, ebedges = get_histogram(sae.enc_b, bins=25)
            dcounts, dedges = get_histogram(sae.dec)
            dbcounts, dbedges = get_histogram(sae.dec_b, bins=25)
            
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
            'dec_b': {
                'counts': dbcounts,
                'edges': dbedges
            }
        }

    def log_loss(self, mse, scaled_mse, l1, loss, batch, latent):
        with torch.no_grad():
            message = {
                'event': 'training_batch', 
                'mse': mse.item(),
                'scaled_mse': scaled_mse.item(),
                'L0': (latent > 0).float().sum(-1).mean().item(),
                'loss': loss.item()
            }

            if l1 is not None:
                message['l1'] = l1.item()

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
