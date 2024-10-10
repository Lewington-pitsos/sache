import json

baseline =        {
    "wandb_project": "vit-sae-multilayer",
    "data_bucket": "sae-activations",
    "log_bucket": "sae-activations",
    "n_feats": 65536,
    "n_tokens": 1_000_000_000,
    "batch_size": 16448 * 2,
    "k": 32,
    "lr": 0.002,
    "d_in": 1024,
    "seq_len": 257,
    "cache_buffer_size": 3,
    "n_cache_workers": 4,
    "batch_norm": False,
    'architecture': 'topk',
    "save_every": 500_000_000,
    "save_checkpoints_to_s3": True,

}

all_configs = []


# for layer in ['11_resid', '14_resid', '17_resid', '20_resid', '22_resid', '2_resid', '5_resid', '8_resid']:
#     clone = baseline.copy()
#     clone['data_name'] = f"multilayer/{layer}"
#     clone['name'] = layer
#     all_configs.append(clone)

# print(f'Generated {len(all_configs)} configs')
# with open('cruft/configs.json', 'w') as f:
#     json.dump(all_configs, f, indent=2)


for layer in ['22_resid']:
    for shuffle in [True, False]:
        for n_experts in [None, 8]:
            clone = baseline.copy()
            clone['data_name'] = f"CLIP-ViT-L-14/{layer}"
            clone['wandb_project'] = 'test-vit-sae-multilayer'
            clone['name'] = 'test-' + layer
            clone['n_tokens'] = 600_000
            clone['shuffle'] = shuffle
            clone['save_every'] = None
            clone['n_tokens'] = 3_000_000
            clone['n_experts'] = n_experts
            all_configs.append(clone)

filename = 'cruft/test_configs.json'
print(f'Generated {len(all_configs)} configs at {filename}')
with open(filename, 'w') as f:
    json.dump(all_configs, f, indent=2)