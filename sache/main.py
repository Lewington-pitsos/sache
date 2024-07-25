from collections import defaultdict
import random
import torch
from sae_lens import HookedSAETransformer, SAE
from multiprocessing import Process

class Generator():
    def __init__(self, transformer_name, cache, device):
        self.cache = cache

        self.transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
        self.dataloader = None

    def _activations_from_batch(self, input_ids):
        _, all_hidden_states = self.transformer.run_with_cache(
            input_ids, 
            prepend_bos=True, 
            stop_at_layer=self.sae.cfg.hook_layer + 1
        )

        return all_hidden_states

    def generate(self):
        for batch in self.dataloader:
            input_ids = batch['input_ids']

            activations = self._activations_from_batch(input_ids)
            self.cache.append(activations)

class Cache():
    def __init__(self):
        self.cache = []

    def __iter__(self):
        idx = 0
        while idx < len(self.cache):
            batch = None
            while batch is None: # keep taking the next batch until one is free
                batch =  self.borrow(idx)
                idx += 1

            yield batch

    def append(self, activations):
        self.cache.append(activations)

    def borrow(self, idx):
        pass

    def give_back(self, idx, activations=None):
        pass

class Shuffler():
    def __init__(self, cache):
        self.cache = cache
        self.shuffle_record = defaultdict(list)

    def choose_two_indices(self):
        a_idx = random.randint(0, len(self.cache) - 1)
        b_idx = random.randint(0, len(self.cache) - 1)

        return a_idx, b_idx

    def _shuffle(self, a, b):
        pass

    def continuous_shuffle(self):
        while True: # while amount of shuffling is less than a certain amount
            self.shuffle()

    def shuffle(self):
        while True:
            a_idx, b_idx = self.choose_two_indices()

            a = self.cache.borrow(a_idx)
            b = self.cache.borrow(b_idx)

            if a is not None and b is not None:
                break

        a, b = self._shuffle(a, b)
        
        self.cache.give_back(a_idx, a)
        self.cache.give_back(b_idx, b)

        self.shuffle_record[a_idx].append(b_idx)
        self.shuffle_record[b_idx].append(a_idx)
    
class Trainer():
    def __init__(self, sae_model, sae_id, cache):
        self.cache = cache

        self.sae, _, _ = SAE.from_pretrained(
            release = sae_model, # see other options in sae_lens/pretrained_saes.yaml
            sae_id = sae_id, 
            device = device
        )

    def _calculate_geometric_median():
        pass

    def _process_batch(self, hidden_states, attention_mask):
        reconstruction = self.sae.forward(hidden_states) * attention_mask.unsqueeze(-1)
        
        rmse = torch.sqrt(torch.mean((hidden_states * attention_mask - reconstruction) ** 2))
        l2 = torch.sqrt(torch.mean(hidden_states ** 2))

        return 

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            for activations, attention_mask in self.cache:
                self._process_batch(activations, attention_mask)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


cache = None
generator = Generator('gpt2', cache, device)

shuffler = Shuffler(cache)

trainer = None # cache


# start new process 
p = Process(target=generator.generate)
p.start()

p = Process(target=shuffler.continuous_shuffle)
p.start()

trainer.train()
