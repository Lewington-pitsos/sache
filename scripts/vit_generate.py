import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import randomname

from sache import vit_generate
from sache import FileDataset

# python scripts/vit_generate.py --run_name "test3" --n_samples=3000000 --batch_size=1024 --batches_per_cache=5 --n_hooks=4 --full_sequence
# python scripts/vit_generate.py --run_name "ViT-3_000_000" --n_samples=3000000 --batch_size=2048
# python scripts/vit_generate.py --run_name "ViT-3mil" --n_samples=3000000 --batch_size=2048

# python scripts/vit_generate.py --run_name "ViT_100_000" --n_samples=100000 --batch_size=2048 --log_every=0

def main(
        run_name=None, 
        bucket_name=None, 
        n_samples=None,
        transformer_name='laion/CLIP-ViT-L-14-laion2B-s32B-b82K', # 24 layers in total
        batch_size=1024,
        log_every=10,
        batches_per_cache=50,
        full_sequence=False,
        n_hooks=None
    ):
    if run_name is None:
        run_name = randomname.generate('adj/', 'n/')
    print('run_name:', run_name)

    data_directory = 'laion/images'
    dataset = FileDataset(root_dir=data_directory)

    hook_locations = [
        {'layer':2, 'module':'resid'},
        {'layer':5, 'module':'resid'},
        {'layer':8, 'module':'resid'},
        {'layer':11, 'module':'resid'},
        {'layer':14, 'module':'resid'},
        {'layer':17, 'module':'resid'},
        {'layer':20, 'module':'resid'},
        {'layer':22, 'module':'resid'},
    ]

    if n_hooks:
        hook_locations = hook_locations[:n_hooks]

    print('number of hooks:', len(hook_locations))

    vit_generate(
        run_name,
        batches_per_cache=batches_per_cache,
        dataset=dataset, 
        transformer_name=transformer_name, 
        batch_size=batch_size, 
        device='cuda',
        hook_locations=hook_locations,
        cache_type='s3_threaded_nonshuffling',
        n_samples=n_samples,
        log_every=None if log_every < 1 else log_every,
        bucket_name=bucket_name,
        full_sequence=full_sequence
    )

if __name__ == '__main__':
    fire.Fire(main)
    
# 641.3362 MB/s @ 8 processes
# 471.1610 MB/s @ 6 processes
# 548.0496 MB/s @ 6 processes memshare