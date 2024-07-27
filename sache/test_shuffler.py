import shutil
import os
import pytest

import torch

from sache.shuffler import ShufflingCache
from sache.cache import WCache

@pytest.fixture
def shuffling_cache():
    cache = WCache('test', save_every=1)

    yield ShufflingCache(cache, buffer_size=4)

    shutil.rmtree(cache.inner_cache_dir)

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
    assert torch.equal(activations, torch.tensor([[0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.],
        [0., 0., 0., 0.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [3., 3., 3., 3.],
        [2., 2., 2., 2.],
        [1., 1., 1., 1.],
        [2., 2., 2., 2.]]))

    all_activations = torch.cat([torch.load(os.path.join(cache_dir, f)) for f in cache_files])
    assert all_activations.shape == (90, 4)
