import torch
import random

class ShufflingCache():
    def __init__(self, cache, buffer_size=8):
        if buffer_size < 2:
            raise ValueError(f'We shuffle by maintaining a buffer of batches and passing in each batch as a random batch sized selection of the buffer. Buffer must be at least 2, got {buffer_size}')

        self.cache = cache
        self.buffer_size = buffer_size
        
        self._buffer = torch.tensor([])
        self._batches_in_buffer = 0

    def _shuffle_buffer(self):
        self._buffer = self._buffer[torch.randperm(len(self._buffer))]

    def append(self, activations):
        self._buffer = torch.cat([self._buffer, activations])
        self._batches_in_buffer += 1

        if self._batches_in_buffer == self.buffer_size:
            self._shuffle_buffer()

            half = len(self._buffer) // 2
            batch_size = len(self._buffer) // self.buffer_size

            for i in range(0, half, batch_size):
                self.cache.append(self._buffer[i:i + batch_size])
            
            self._buffer = self._buffer[half:]
            self._batches_in_buffer = self.buffer_size // 2

    def finalize(self):
        if len(self._buffer) > 0:
            self._shuffle_buffer()
            
            batch_size = len(self._buffer) // self._batches_in_buffer
            for i in range(0, len(self._buffer), batch_size):
                self.cache.append(self._buffer[i:i + batch_size])

            self._buffer = torch.tensor([])
            self._batches_in_buffer = 0

        self.cache.finalize()

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
            