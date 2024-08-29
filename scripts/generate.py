import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import randomname

from sache.generator import generate

def main(
        run_name=None, 
        bucket_name=None, 
        dataset_name='Skylion007/openwebtext', 
        n_samples=300_000, 
        hook_name='blocks.10.hook_resid_post', 
        transformer_name='gpt2',
        max_length=1024,
    ):
    if run_name is None:
        run_name = randomname.generate('adj/', 'n/')
    print('run_name:', run_name)


    dataset = load_dataset(dataset_name, trust_remote_code=True)['train'].select(range(n_samples))
    print('dataset loaded')

    generate(
        run_name,
        batches_per_cache=128,
        dataset=dataset, 
        transformer_name=transformer_name, 
        max_length=max_length, 
        batch_size=8, 
        text_column_name='text', 
        device='cuda',
        layer=11,
        cache_type='s3_threaded_nonshuffling',
        hook_name=hook_name,
        log_every=100,
        bucket_name=bucket_name
    )

if __name__ == '__main__':
    fire.Fire(main)
    
