import torch
from datasets import load_dataset
from sae_lens import HookedSAETransformer
from torch.utils.data import DataLoader 

from sache.cache import Cache
from sache.tok import tokenize_and_concatenate


def activations_from_batch(transformer, input_ids, layer):
    _, all_hidden_states = transformer.run_with_cache(
        input_ids, 
        prepend_bos=True, 
        stop_at_layer=layer + 1
    )

    return all_hidden_states

    def generate(self):
        for batch in self.dataloader:
            input_ids = batch['input_ids']

            activations = self._activations_from_batch(input_ids)
            self.cache.append(activations)

def generate(
        cache_dir, 
        dataset, 
        transformer_name, 
        max_length, 
        batch_size, 
        text_column_name, 
        device,
        output_layer,
    ):
    dataset = load_dataset(dataset)
    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)


    dataset = tokenize_and_concatenate(
        dataset, 
        transformer.tokenizer, 
        streaming=True, 
        max_length=max_length, 
        column_name=text_column_name, 
        add_bos_token=True
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cache = Cache(cache_dir)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        activations = activations_from_batch(transformer, input_ids, output_layer)

        attention_mask = attention_mask.unsqueeze(-1).expand_as(activations)
        activations = torch.cat([activations, attention_mask], dim=-1)

        cache.append(activations)

