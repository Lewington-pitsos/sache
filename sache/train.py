import json
import torch

from sache.cache import S3RBatchingCache

class SAE(torch.nn.Module):
    def __init__(self, n_features, hidden_size):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.rand(hidden_size, n_features))
        self.dec = torch.nn.Parameter(torch.rand(n_features, hidden_size))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        features = self.activation(x @ self.enc)
        return features @ self.dec, features

def build_cache(run_name, batch_size):
    with open('.credentials.json') as f:
        credentials = json.load(f)

    cache = S3RBatchingCache.from_credentials(
        batch_size=batch_size,
        access_key_id=credentials['AWS_ACCESS_KEY_ID'], 
        secret=credentials['AWS_SECRET'], 
        local_cache_dir='cache/' + run_name, 
        s3_prefix=run_name,
    )
    return cache

def train(run_name, hidden_size, n_features, batch_size=32):
    cache = build_cache(run_name, batch_size=batch_size)
    sae = SAE(n_features=n_features, hidden_size=hidden_size)

    n_batches = 10
    for i, activations in enumerate(cache):
        activations, _, = activations[:, :, :hidden_size], activations[:, :, hidden_size]

        reconstruction, _ = sae(activations)

        rmse = torch.sqrt(torch.mean((activations - reconstruction) ** 2))
        
        print(rmse)

        if i > n_batches:
            break

if __name__ == '__main__':
    train('thundering-barracuda/', 768, 384)