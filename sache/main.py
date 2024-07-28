from datasets import load_dataset
import randomname

from sache.generator import generate

for cache_type in ['s3_threaded']:
    run_name = randomname.generate('adj/', 'n/', 'n/') + '-' + cache_type

    print('run_name:', run_name)

    dataset = load_dataset('NeelNanda/pile-10k')['train'].select(range(250))

    generate(
        run_name,
        dataset=dataset, 
        batches_per_cache=16,
        transformer_name='gpt2', 
        max_length=768, 
        batch_size=16, 
        text_column_name='text', 
        device='cuda',
        layer=10,
        cache_type='s3_threaded',
        hook_name='blocks.9.hook_resid_post',
    )

