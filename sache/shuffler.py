import torch

from sache.cache import WCache

class ShufflingWCache():
    @classmethod
    def from_params(cls, buffer_size=8, *args, **kwargs):
        cache = WCache(*args, **kwargs)
        return ShufflingWCache(cache, buffer_size)

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

    def save_mean_std(self, mean, std):
        self.cache.save_mean_std(mean, std)

    def finalize(self):
        if len(self._buffer) > 0:
            self._shuffle_buffer()
            
            batch_size = len(self._buffer) // self._batches_in_buffer
            for i in range(0, len(self._buffer), batch_size):
                self.cache.append(self._buffer[i:i + batch_size])

            self._buffer = torch.tensor([])
            self._batches_in_buffer = 0

        self.cache.finalize()            