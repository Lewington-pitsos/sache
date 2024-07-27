import torch
import random
from sache.cache import WCache

class ShufflingCache(WCache):
    def __init__(self, buffer_size=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if buffer_size < 2:
            raise ValueError(f'We shuffle by maintaining a buffer of batches and passing in each batch as a random batch sized selection of the buffer. Buffer must be at least 2, got {buffer_size}')

        self.buffer = torch.tensor([])

    def _shuffle_buffer(self):
        self.buffer = self.buffer[torch.randperm(len(self.buffer))]

    def append(self, activations):
        self.buffer = torch.cat([self.buffer, activations])

        if len(self.buffer) >= self.buffer_size:
            self._shuffle_buffer()

            half = len(self.buffer) // 2

            batch_size = self.buffer_size // 2

            for i in range(0, half, batch_size):
                super().append(self.buffer[i:i + batch_size])
            
            self.buffer = self.buffer[half:]

    def finalize(self):
        if len(self.buffer) > 0:
            self._shuffle_buffer()
            super().append(self.buffer)
            self.buffer = torch.tensor([])
        super().finalize()

class Shuffler():
    def __init__(self, cache, n_shuffles):
        self.cache = cache
        self.n_shuffles = n_shuffles

    def choose_two_indices(self):
        a_idx = random.randint(0, len(self.cache) - 1)
        b_idx = random.randint(0, len(self.cache) - 1)

        return a_idx, b_idx

    def shuffle_in_place(self, activations, loaded):
        pass

    def shuffle(self, id):
        activations = self.cache.take(id)

        loaded_id, loaded = self.cache.take_random()
        self.shuffle_in_place(activations, loaded)

        self.cache.give_back(loaded_id, loaded)
        self.cache.give_back(id, activations)
            