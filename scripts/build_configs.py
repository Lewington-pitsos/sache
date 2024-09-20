import json

baseline =     {
        "name": "k32-baseline",
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
        "n_experts": None,
        "cache_buffer_size": 10,
        "n_cache_workers": 6,
        "architecture": 'topk',
        "batch_norm": False,
        "use_wandb": False
    }

    
all_configs = []

for batch_expansion in [8]:
    config = baseline.copy()
    config['name'] = f"kk32-baseline-{batch_expansion}"
    config['batch_size'] = 1024 * batch_expansion
    all_configs.append(config)

    for n_experts in [4, 16, 64, 256]:
        config = baseline.copy()
        config['name'] = f"kk32-expansion-{batch_expansion}-experts-{n_experts}"
        config['n_experts'] = n_experts
        config['batch_size'] = 1024 * batch_expansion
        all_configs.append(config)


print('num configs', len(all_configs))
with open('cruft/switch_configs.json', 'w') as f:
    json.dump(all_configs, f, indent=2)