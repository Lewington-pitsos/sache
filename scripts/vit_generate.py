import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import randomname

from sache.generator import vit_generate
from sache.imgloader import FileDataset

# python scripts/vit_generate.py --run_name "ViT-3_000_000" --n_samples=3000000 --batch_size=2048
# python scripts/vit_generate.py --run_name "ViT-45_000_000" --n_samples=46000000 --batch_size=2048

# python scripts/vit_generate.py --run_name "ViT_100_000" --n_samples=100000 --batch_size=2048 --log_every=0

def main(
        run_name=None, 
        bucket_name=None, 
        n_samples=None,
        transformer_name='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        batch_size=1024,
        log_every=10,
    ):
    if run_name is None:
        run_name = randomname.generate('adj/', 'n/')
    print('run_name:', run_name)

    data_directory = 'images'
    dataset = FileDataset(root_dir=data_directory)

    vit_generate(
        run_name,
        batches_per_cache=50,
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
    )

if __name__ == '__main__':
    fire.Fire(main)
    
