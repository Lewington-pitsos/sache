import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import randomname

from sache import vit_generate
from sache import FileDataset

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
        full_sequence=False
    ):
    if run_name is None:
        run_name = randomname.generate('adj/', 'n/')
    print('run_name:', run_name)

    data_directory = 'images'
    dataset = FileDataset(root_dir=data_directory)

    vit_generate(
        run_name,
        batches_per_cache=batches_per_cache,
        dataset=dataset, 
        transformer_name=transformer_name, 
        batch_size=batch_size, 
        device='cuda',
        layer=-2,
        hook_name="resid",
        cache_type='s3_threaded_nonshuffling',
        n_samples=n_samples,
        log_every=None if log_every < 1 else log_every,
        bucket_name=bucket_name,
        full_sequence=full_sequence
    )

if __name__ == '__main__':
    fire.Fire(main)
    
