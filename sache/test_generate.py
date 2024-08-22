import json
import os
from datasets import load_dataset

from sache.generator import generate
from sache.cache import OUTER_CACHE_DIR, INNER_CACHE_DIR


import time


def test_generate():
    dataset = load_dataset('NeelNanda/pile-10k')['train'].select(range(16))
    
    human_readable_time = 'test-' + time.strftime("%Y%m%d-%H%M%S")

    generate(
        run_name=human_readable_time,
        dataset=dataset, 
        batches_per_cache=2,
        transformer_name='EleutherAI/pythia-70m', 
        max_length=64, 
        batch_size=2, 
        text_column_name='text', 
        device='cpu',
        layer=3,
        hook_name='blocks.2.hook_resid_post',
        cache_type='local',
        log_every=None,
        num_proc=4,
    )

    full_cache_dir = os.path.join(OUTER_CACHE_DIR, human_readable_time, INNER_CACHE_DIR)
    cache_files = os.listdir(full_cache_dir)
    assert len(cache_files) == 38

    with open(os.path.join(OUTER_CACHE_DIR, human_readable_time, 'metadata.json')) as f:
        metadata = json.load(f)

    assert 'mean' in metadata
    assert 'std' in metadata
    assert len(metadata['mean']) == 512
    assert len(metadata['std']) == 512

    for file in cache_files:
        os.remove(os.path.join(full_cache_dir, file))
    os.rmdir(full_cache_dir)

