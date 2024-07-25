import random
import os
import torch
from uuid import uuid4

STAGES = ['saving', 'saved', 'shuffled']

class Cache():
    def __init__(self, cache_dir, loan_stage, return_stage, device=None):
        self.cache_dir = cache_dir
        self._relevant = {}
        self.loan_stage = loan_stage
        self.return_stage = return_stage
        self.device = device

        if loan_stage not in STAGES:
            raise ValueError(f'loan_stage must be among {STAGES} got {loan_stage}')

        if return_stage not in STAGES:
            raise ValueError(f'return_stage must be among {STAGES} got {return_stage}')

    def __iter__(self):
        self.reset_buffer(self.loan_stage)
        for id, v in self._relevant.items():
            yield id, torch.load(v, map_location=self.device)

    def __len__(self):
        return len(self._relevant)
    
    def loaned_keys(self):
        return self._relevant.keys()

    def _load(self, id):
        return torch.load(self._relevant[id], map_location=self.device)

    def give_back(self, id, activations):
        self.save(activations, self.return_stage, id)

    def take(self, id):
        activations = self._load(id)

        del self._relevant[id]

        return activations

    def take_random(self):
        ids = list(self._relevant.keys())
        id = random.choice(ids)

        return id, self._load(id)

    def _filename(self, id, stage):
        return os.path.join(self.cache_dir, f'{id}.{stage}.pt')

    def append(self, activations):
        self.save(activations, 'saved', str(uuid4()))

    def save(self, activations, stage, id):
        filename = self._filename(id, 'saving')
        relative_path = os.path.join(self.cache_dir, filename)
        torch.save(relative_path, activations)

        os.rename(relative_path, relative_path.replace('.saving.pt', f'.{stage}.pt')) # prevent reads of incomplete files

    def reset_buffer(self):
        all_files = os.listdir(self.cache_dir)
        self._n_files = len(all_files)

        self._relevant = {}
        for file in all_files:
            id = file.split('.')[0]
            if file.endswith(f'.{self.loan_stage}.pt'):
                self._relevant[id] = os.path.join(self.cache_dir, file)


