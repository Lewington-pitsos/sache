import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch

from sache.cache import RCache


data_dir = 'cruft/rcache_test'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

n_files = 8
gb = 3
if len(os.listdir(data_dir)) == 0:
    for i in range(n_files):
        tensor_size = gb * 1024 * 1024 * 1024
        tensor_shape = (tensor_size // 4,)  # Using float32, so 4 bytes per element

        tensor = torch.randn(tensor_shape, dtype=torch.float32)

        torch.save(tensor, os.path.join(data_dir, f'{i}.pt'))

print('data created, building cache')

start = time.time()
cache = RCache(data_dir, device='cpu', num_workers=4)

for _ in cache:
    pass

end = time.time()

seconds = round(end - start, 2)
print(f'Duration, {seconds}')
print(f'Files per second: {n_files / seconds}')
print(f'GB per second: {n_files / seconds * gb}')

