import time
import shutil
import os
import pytest

import torch

from sache.shuffler import ShufflingCache
from sache.cache import WCache, ThreadedCache


@pytest.fixture
def shuffling_cache():
    cache = WCache('test', save_every=1)

    yield ShufflingCache(cache, buffer_size=4)

    shutil.rmtree(cache.outer_cache_dir)

def test_threaded_shuffling_cache(shuffling_cache):
    torch.manual_seed(0)
    cache = ThreadedCache(shuffling_cache)
    for i in range(9):
        cache.append(torch.tensor([[i, i, i, i]] * 10))

    cache.finalize()

    time.sleep(0.5)

    cache_dir = cache.cache.cache.cache_dir

    cache_files = os.listdir(cache_dir)
    assert len(cache_files) == 9 

    file_modified_first = min(cache_files, key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)))
    activations = torch.load(os.path.join(cache_dir, file_modified_first))

    assert activations.shape == (10, 4)
    print(activations)
    assert torch.min(activations) != torch.max(activations)

    all_activations = torch.cat([torch.load(os.path.join(cache_dir, f)) for f in cache_files])
    assert all_activations.shape == (90, 4)

def test_shuffling_cache_shuffles(shuffling_cache):
    torch.manual_seed(0)
    for i in range(9):
        shuffling_cache.append(torch.tensor([[i, i, i, i]] * 10))

    shuffling_cache.finalize()

    cache_dir = shuffling_cache.cache.cache_dir

    cache_files = os.listdir(cache_dir)
    assert len(cache_files) == 9 

    file_modified_first = min(cache_files, key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)))
    activations = torch.load(os.path.join(cache_dir, file_modified_first))

    assert activations.shape == (10, 4)
    print(activations)
    assert torch.min(activations) != torch.max(activations)

    all_activations = torch.cat([torch.load(os.path.join(cache_dir, f)) for f in cache_files])
    assert all_activations.shape == (90, 4)
