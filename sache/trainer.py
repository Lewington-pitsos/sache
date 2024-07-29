import torch

from sache.cache import Cache


class SAE(torch.nn.Module):
    def __init__(self, n_features, hidden_size):
        super(SAE, self).__init__()
        self.enc = torch.nn.Parameter(torch.rand(hidden_size, n_features))
        self.dec = torch.nn.Parameter(torch.rand(n_features, hidden_size))

    def forward(self, x):
        features = x @ self.enc
        return features @ self.dec, features

def train(cache_dir, hidden_size, n_features, epochs):
    cache = Cache(cache_dir)
    sae = SAE(n_features=n_features, hidden_size=hidden_size)

    for epoch in epochs:
        for _, activations in cache:
            attention_mask, activations = activations[:, -1], activations[:, :-1]

            reconstruction, _ = sae(activations) * attention_mask.unsqueeze(-1)

            rmse = torch.sqrt(torch.mean((activations * attention_mask - reconstruction) ** 2))
            
            print(rmse)