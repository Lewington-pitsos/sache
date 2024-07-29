from datasets import load_dataset
import randomname

from sache.generator import generate


run_name = randomname.generate('adj/', 'n/')

print('run_name:', run_name)

dataset = load_dataset('Skylion007/openwebtext', trust_remote_code=True)['train'].select(range(300_000))

print('dataset loaded')

generate(
    run_name,
    batches_per_cache=128,
    dataset=dataset, 
    transformer_name='gpt2', 
    max_length=1024, 
    batch_size=8, 
    text_column_name='text', 
    device='cuda',
    layer=10,
    cache_type='local_threaded',
    hook_name='blocks.9.hook_resid_post',
    log_every=10
)

