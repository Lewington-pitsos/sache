import json

baseline =        {
    "wandb_project": "vit-switch-sae",
    "n_feats": 65536,
    "n_tokens": 24_000_000,
    "batch_size": 4024 * 2,
    "k": 32,
    "lr": 0.0008,
    "l1_coefficient": 0.00008,
    'data_name': "ViT-45_000_000",
    "d_in": 1024,
    "samples_per_file": 102_400,
    "seq_len": 1,
    "cache_buffer_size": 3,
    "n_cache_workers": 4,
    "batch_norm": False,
    "save_every": 10_000_000
}

all_configs = []

for l1 in [ 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 8e-05, 9e-05, 0.00012, 0.00016, 0.0002]:
    config = baseline.copy()
    config['l1_coefficient'] = l1
    config['architecture'] = 'relu'
    config['name'] = f'relu-l1-{l1}'
    all_configs.append(config)

for k in [8, 16, 32, 64,	128, 256]:
    config = baseline.copy()
    config['k'] = k
    config['architecture'] = 'topk'
    for n_experts in [None, 8, 16, 32, 64, 128]:
        config['n_experts'] = n_experts
        config['name'] = f'topkk-{k}-experts-{n_experts}'
        all_configs.append(config)


print(f'Generated {len(all_configs)} configs')
with open('cruft/switch_configs.json', 'w') as f:
    json.dump(all_configs, f, indent=2)