from sae_lens import HookedSAETransformer

from sache.cache import Cache

class Generator():
    def __init__(self, cache, transformer_name, dataset_path, device):
        self.cache = cache
        self.dataset_path = dataset_path

        self.transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
        self.dataloader = None

    def _activations_from_batch(self, input_ids):
        _, all_hidden_states = self.transformer.run_with_cache(
            input_ids, 
            prepend_bos=True, 
            stop_at_layer=self.sae.cfg.hook_layer + 1
        )

        return all_hidden_states

    def generate(self):
        for batch in self.dataloader:
            input_ids = batch['input_ids']

            activations = self._activations_from_batch(input_ids)
            self.cache.append(activations)

def generate(cache_dir, **kwargs):
    cache = Cache(cache_dir)
    generator = Generator(cache, **kwargs)
    generator.generate()

