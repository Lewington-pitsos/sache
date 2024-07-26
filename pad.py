from sae_lens import HookedSAETransformer
import torch
transformer_name = 'EleutherAI/pythia-70m'
transformer = HookedSAETransformer.from_pretrained(transformer_name)

input_ids = torch.randint(0, 50256, (4, 512))

_, activations = transformer.run_with_cache(
    input_ids, 
    prepend_bos=False, # each sample is actually multiple concatenated samples
    stop_at_layer=5
)

print(activations.keys())

print(activations['blocks.4.hook_resid_post'])


