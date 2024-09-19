import json

baseline =        {
    "wandb_project": "vit-sae-test",
    "n_feats": 65536,
    "batch_size": 1024,
    "k": 32,
    "lr": 0.0004,
    "l1_coefficient":0.00008,
    'data_name': "ViT-3_000_000",
    "d_in": 1024,
    "samples_per_file": 20480,
    "seq_len": 1,
    "cache_buffer_size": 10,
    "n_cache_workers": 6,
    "batch_norm": False,
}

all_configs = []
for l1 in [0.0256, 0.0128, 0.0064,	0.0032,	0.0016,	0.0008,	0.0004]:
    config = baseline.copy()
    config['l1_coefficient'] = l1
    config['architecture'] = 'relu'
    config['name'] = f'relu-l1-{l1}'
    all_configs.append(config)


for k in [8,	16,	32,	64,	128,	256]:
    config = baseline.copy()
    config['k'] = k
    config['architecture'] = 'topk'
    for n_experts in [None, 8, 16, 32, 64, 128]:
        config['n_experts'] = n_experts
        config['name'] = f'topkk-{k}-experts-{n_experts}'
        all_configs.append(config)

    for n_experts in [2, 4, 8]:
        config['n_experts'] = n_experts
        config['n_feats'] = config['n_feats'] * n_experts
        config['name'] = f'topkk-{k}-experts-{n_experts}-flopscaled'
        all_configs.append(config)

with open('cruft/switch_configs.json', 'w') as f:
    json.dump(all_configs, f)