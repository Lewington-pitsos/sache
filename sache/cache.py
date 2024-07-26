import random
import os
import torch
from uuid import uuid4

STAGES = ['saving', 'saved', 'shuffled']

class Cache():
    def __init__(self, cache_dir, read_stage, device=None):
        self.cache_dir = cache_dir
        self._relevant = {}
        self.loan_stage = read_stage
        self.device = device

        if read_stage not in STAGES:
            raise ValueError(f'loan_stage must be among {STAGES} got {read_stage}')

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
        self.save(activations, self.loan_stage, id)

    def take(self, id):
        activations = self._load(id)

        del self._relevant[id]

        return activations
    
    def move_stage(self, id, current_stage, next_stage):
        old_filename = self._filename(id, current_stage)
        new_filename = self._filename(id, next_stage)
        os.rename(old_filename, new_filename)

    def take_random(self):
        ids = list(self._relevant.keys())
        id = random.choice(ids)

        return id, self._load(id)

    def _filename(self, id, stage):
        return os.path.join(self.cache_dir, f'{id}.{stage}.pt')

    def append(self, activations, attention_mask=None):
        self.save(activations, 'saved', str(uuid4()))

    def save(self, activations, stage, id):
        filename = self._filename(id, 'saving')
        relative_path = os.path.join(self.cache_dir, filename)
        torch.save(relative_path, activations)

        self.move_stage(id, 'saving', stage)
        
    def reset_buffer(self):
        all_files = os.listdir(self.cache_dir)
        self._n_files = len(all_files)

        self._relevant = {}
        for file in all_files:
            id = file.split('.')[0]
            if file.endswith(f'.{self.loan_stage}.pt'):
                self._relevant[id] = os.path.join(self.cache_dir, file)


