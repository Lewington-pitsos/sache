import os
import time
from collections import defaultdict
import random
import torch
from sae_lens import HookedSAETransformer
from multiprocessing import Process
import randomname


class Cache():
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache = []
        self.logger = ProcessLogger()

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

class Generator():
    def __init__(self, cache, transformer_name, dataset_path, device):
        self.cache = cache
        self.logger = ProcessLogger()
        self.dataset_path = dataset_path

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

def generate(cache_dir, **kwargs):
    cache = Cache(cache_dir)
    generator = Generator(cache, **kwargs)
    generator.generate()

class Shuffler():
    def __init__(self, cache):
        self.cache = cache
        self.shuffle_record = defaultdict(list)
        self.logger = ProcessLogger()

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
    
def shuffle(cache_dir):
    cache = Cache(cache_dir)
    shuffler = Shuffler(cache)
    shuffler.continuous_shuffle()

class Trainer():
    def __init__(self, sae, cache, n_epochs):
        self.cache = cache  

        self.sae = sae
        self.logger = ProcessLogger()
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
        while not self._enough_of_a_head_start(cache):
            time.sleep(1)

        for epoch in range(self.n_epochs):
            for activations, attention_mask in self.cache:
                self._process_batch(activations, attention_mask)

def train(cache_dir, **kwargs):
    cache = Cache(cache_dir)
    sae = None
    trainer = Trainer(sae, cache, **kwargs)
    trainer.train()

class ProcessLogger():
    def __init__(self):
        pass

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


params = {
    'epochs': 10,
    'transformer_name': 'gpt2',
    'dataset_path': 'NeelNanda/pile-10k'
}

run_name = randomname.generate('adj/', 'n/', 'n/')
base_cache_dir = 'cache'
cache_dir = os.path.join(base_cache_dir, run_name)

cache = Cache(cache_dir=cache_dir)

generate_process = Process(target=generate, kwargs=({'cache_dir': cache_dir, **params}))
generate_process.start()

shuffle_process = Process(target=shuffle, kwargs=({'cache_dir': cache_dir}))
shuffle_process.start()

train(cache_dir, **params)

generate_process.join()
shuffle_process.join()


