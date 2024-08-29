from sae_lens import HookedSAETransformer
import torch
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import datasets
from collections import defaultdict
import time
import os
import json

def get_texts(n_texts):
    data = datasets.load_dataset('NeelNanda/pile-10k')['train']
    texts = []

    for i, t in enumerate(data):
        if len(t['text']) > 6000:
            texts.append(t['text'])

        if len(texts) == n_texts:
            break
    
    if len(texts) < n_texts:
        raise ValueError('not enough texts, only found', len(texts))


    print('went through', i, 'texts')

    return texts

def get_inputs(tok, texts, max_length=1024):
    input = tok(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    assert torch.all(input.attention_mask == 1)
    return input

def generate_unigrams(transformer, layer, hook_name, all_tokens, t_fn, device, batch_size=32):
    all_tokens = list(all_tokens)

    sequences = [t_fn(t) for t in all_tokens]

    t_acts = None
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        ids = torch.tensor(batch, device=device)

        _, activations = transformer.run_with_cache(ids, prepend_bos=False, stop_at_layer=layer)
        activations = activations[hook_name]
        if t_acts is None:
            t_acts = activations[:, -1, :].to('cpu')
        else:
            t_acts = torch.concat((t_acts, activations[:, -1, :].to('cpu')), dim=0)

    act_dict = {}

    for i, act in enumerate(t_acts):
        act_dict[all_tokens[i]] = act

    return act_dict
    
def get_distances(sequence_acts, input_ids, token_dict, device):
    unigram_acts = torch.empty_like(sequence_acts)

    for i, id in enumerate(input_ids):
        unigram_act = token_dict[int(id)].to(device)
        unigram_acts[i] = unigram_act

    cos = cosine_similarity(sequence_acts, unigram_acts, dim=-1).cpu().numpy().tolist()

    return cos



def get_sequence_activations(transformer, layer, hook_name, input, full_sequence_batch_size, device):
    sequence_acts = []

    with torch.no_grad():
        for i in range(0, input.input_ids.shape[0], full_sequence_batch_size):
            batch = input.input_ids[i:i+full_sequence_batch_size].to(device)
            acts = transformer.run_with_cache(batch, prepend_bos=False, stop_at_layer=layer)[1][hook_name]

            for j in range(batch.shape[0]):
                sequence_acts.append(acts[j].to('cpu'))

    return sequence_acts

def layer_distance(transformer, layer, hook_name, sequence_ids, unigram_fns, full_sequence_batch_size, token_sequence_batch_size, device):
    with torch.no_grad():
        start = time.time()
        n_texts = sequence_ids.input_ids.shape[0]
        sequence_acts = get_sequence_activations(transformer, layer, hook_name, sequence_ids, full_sequence_batch_size, device)

        print('generated sequence activations in', time.time() - start)

        all_tokens = set()
        for i in range(n_texts):
            all_tokens.update(sequence_ids.input_ids[i].tolist())

        unigrams = {}

        for name, fn in unigram_fns.items():
            unigrams[name] = generate_unigrams(transformer, layer, hook_name, all_tokens, fn, device, batch_size=token_sequence_batch_size)

        print('generated unigrams in', time.time() - start)

        coss = defaultdict(list)
        for i in range(n_texts):
            text_activations = sequence_acts[i].squeeze()
            text_ids = sequence_ids.input_ids[i]

            for k, token_dict in unigrams.items():
                cos = get_distances(text_activations.to(device), text_ids, token_dict, device)
                coss[k].append(cos)


    print('generated correlations in', time.time() - start)

    return coss


def save_model_correlations(transformer_name, texts, layers, device, unigram_fns, full_sequence_batch_size, token_sequence_batch_size, prefix):
    transformer = HookedSAETransformer.from_pretrained(transformer_name, device=device)
    transformer.eval()
    tok = transformer.tokenizer
    input = get_inputs(tok, texts)

    for required_layer, hook_name in layers:
        filename = f'../cruft/{prefix}_{transformer_name}_{hook_name}_cos.json'
        if not os.path.exists(filename):
            
            cos = layer_distance(transformer, required_layer, hook_name, input, unigram_fns, full_sequence_batch_size, token_sequence_batch_size, device)

            if not os.path.exists('../cruft'):
                os.makedirs('../cruft')

            with open(filename, 'w') as f:
                json.dump(cos, f)


with open('../.credentials.json') as f:
    creds = json.load(f)

os.environ['HF_TOKEN'] = creds['HF_TOKEN']

n_texts = 50
device = 'cuda'

lgpt2 = []

prefix=str(n_texts)

for i in range(11, -1, -1):
    hook_name = f'blocks.{i}.hook_resid_pre'
    layer = i+1
    lgpt2.append((layer, hook_name))

gpt2params = {
    'prefix': prefix,

    'transformer_name': 'gpt2-small',
    'layers': lgpt2,
    'device': device,
    'full_sequence_batch_size': 8,
    'token_sequence_batch_size': 2048,
    'unigram_fns': {
        '[t]': lambda t: [t],
        '[<bos>, t]':     lambda t: [50256, t],
        '[<bos>, t, !]':     lambda t: [50256, t, 0], 
        '[t, t]':  lambda t: [t, t], 
        '[" ", t]':   lambda t: [220, t],  
        '[" ", " ", t]':  lambda t: [220, 220, t],  
        '[37233, t]':    lambda t: [37233, t],  
        '[<bos>, 37233]': lambda t: [50256, 37233], 
    }
}

lgemma2 = []

for i in range(17, -1, -1):
    hook_name = f'blocks.{i}.hook_resid_pre'
    layer = i+1
    lgemma2.append((layer, hook_name))

gemma_ugram_fns =  {
    '[t]': lambda t: [t],
    '[<bos>, t]':     lambda t: [2, t], 
    '[<bos>, t, !]':     lambda t: [2, t, 235341], 
    '[<pad>, <bos>, t]':     lambda t: [0, 2, t],
    '[<bos>, <pad>, t]':     lambda t: [2, 0, t],
    '[<pad>, t]':     lambda t: [0, t],
    '[t, t]':  lambda t: [t, t], 
    '[37233, t]':    lambda t: [37233, t],  
    '[" ", t]':   lambda t: [139, t],  # 139 is s space in gemma
    '[<bos>, 37233]': lambda t: [2, 37233], 

}
gemma2_2b = {
    'prefix': prefix,

    'transformer_name': 'gemma-2b',
    'layers': lgemma2,
    'device': device,
    'full_sequence_batch_size': 2,
    'token_sequence_batch_size': 512,
    'unigram_fns': gemma_ugram_fns
}

lgemma2_9b = []

for i in range(40, -1, -3):
    hook_name = f'blocks.{i}.hook_resid_pre'
    layer = i+1
    lgemma2_9b.append((layer, hook_name))

gemma2_9b = {
    'prefix': prefix,
    'transformer_name': 'gemma-2-9b',
    'layers': lgemma2_9b,
    'device': device,
    'full_sequence_batch_size': 1,
    'token_sequence_batch_size': 512,
    'unigram_fns': gemma_ugram_fns
}

params = [
    gemma2_9b,
    gemma2_2b,  
    gpt2params
]


texts = get_texts(n_texts)

for p in params:
    print(p)
    save_model_correlations(texts=texts, **p)