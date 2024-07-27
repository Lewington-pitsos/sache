from datasets import load_dataset
import randomname

from sache.generator import generate

run_name = randomname.generate('adj/', 'n/', 'n/')

print('run_name:', run_name)

dataset = load_dataset('NeelNanda/pile-10k')['train']

generate(
    run_name,
    dataset=dataset, 
    batches_per_cache=2,
    transformer_name='gpt2', 
    max_length=768, 
    batch_size=8, 
    text_column_name='text', 
    device='cuda',
    layer=9,
    hook_name='blocks.9.hook_resid_post',
    batches_per_cache=32
)

