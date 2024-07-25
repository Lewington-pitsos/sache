import torch
import time

from sache.cache import Cache

class Trainer():
    def __init__(self, sae, cache, n_epochs):
        self.cache = cache  

        self.sae = sae
        self.n_epochs = n_epochs


    def _calculate_geometric_median():
        pass

    def _process_batch(self, hidden_states, attention_mask):
        reconstruction = self.sae.forward(hidden_states) * attention_mask.unsqueeze(-1)
        
        rmse = torch.sqrt(torch.mean((hidden_states * attention_mask - reconstruction) ** 2))
        l2 = torch.sqrt(torch.mean(hidden_states ** 2))

        return 

    def _enough_of_a_head_start(self, cache):
        return len(cache) > 100

    def train(self):
        while not self._enough_of_a_head_start(self.cache):
            time.sleep(1)

        for epoch in range(self.n_epochs):
            for activations, attention_mask in self.cache:
                self._process_batch(activations, attention_mask)

def train(cache_dir, **kwargs):
    cache = Cache(cache_dir)
    sae = None
    trainer = Trainer(sae, cache, **kwargs)
    trainer.train()

