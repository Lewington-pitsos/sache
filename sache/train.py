import json
import torch

from sache.cache import S3RCache


class SAE(torch.nn.Module):
    def __init__(self, n_features, hidden_size):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.rand(hidden_size, n_features))
        self.dec = torch.nn.Parameter(torch.rand(n_features, hidden_size))

    def forward(self, x):
        features = x @ self.enc
        return features @ self.dec, features

def build_cache(run_name, max_gb=100):
    with open('.credentials.json') as f:
        credentials = json.load(f)

    cache = S3RCache.from_credentials(
        credentials['AWS_ACCESS_KEY_ID'], 
        credentials['AWS_SECRET'], 
        local_cache_dir='cache/' + run_name, 
        s3_prefix=run_name,
        max_gb=max_gb
    )
    return cache

def train(run_name, hidden_size, n_features, epochs):
    cache = build_cache(run_name)
    sae = SAE(n_features=n_features, hidden_size=hidden_size)

    for epoch in range(epochs):
        for _, activations in cache:
            attention_mask, activations = activations[:, -1], activations[:, :-1]

            reconstruction, _ = sae(activations) * attention_mask.unsqueeze(-1)

            rmse = torch.sqrt(torch.mean((activations * attention_mask - reconstruction) ** 2))
            
            print(rmse)