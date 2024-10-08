import json

baseline =        {
    "wandb_project": "vit-sae-multilayer",
    "n_feats": 65536,
    "n_tokens": 5_000_000_000,
    "batch_size": 16384,
    "k": 32,
    "lr": 0.001,
    "d_in": 1024,
    "seq_len": 257,
    "cache_buffer_size": 3,
    "n_cache_workers": 4,
    "batch_norm": False,
    'architecture': 'topk',
    "save_every": 500_000_000
}

all_configs = []


for layer in ['11_resid', '14_resid', '17_resid', '20_resid', '22_resid', '2_resid', '5_resid', '8_resid']:
    clone = baseline.copy()
    clone['data_name'] = f"multilayer/{layer}"
    clone['name'] = layer
    all_configs.append(clone)

print(f'Generated {len(all_configs)} configs')
with open('cruft/flop_switch_configs.json', 'w') as f:
    json.dump(all_configs, f, indent=2)