import json
import torch
from tqdm import tqdm

from sache.cache import S3RBatchingCache
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

def build_cache(run_name, batch_size, device):
    with open('.credentials.json') as f:
        credentials = json.load(f)

    cache = S3RBatchingCache.from_credentials(
        batch_size=batch_size,
        access_key_id=credentials['AWS_ACCESS_KEY_ID'], 
        secret=credentials['AWS_SECRET'], 
        local_cache_dir='cache/' + run_name, 
        s3_prefix=run_name + '/',
        device=device
    )
    return cache

def train(run_name, hidden_size, n_features, device, batch_size=32):
    logger = ProcessLogger(run_name)
    cache = build_cache(run_name, batch_size=batch_size, device=device)
    sae = SAE(n_features=n_features, hidden_size=hidden_size, device=device)

    n_batches = 10_000

    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for i, activations in tqdm(enumerate(cache), total=n_batches):
        optimizer.zero_grad()
        activations, _, = activations[:, :, :hidden_size], activations[:, :, hidden_size]

        reconstruction, _ = sae(activations)

        rmse = torch.sqrt(torch.mean((activations - reconstruction) ** 2))
        
        rmse.backward()
        optimizer.step()

        logger.log({'event': 'training_batch', 'rmse': rmse.item()})    

        if i > n_batches:
            break

if __name__ == '__main__':
    train('active-camera', 768, 384, device='cuda', batch_size=256)