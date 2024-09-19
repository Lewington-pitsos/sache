import json

baseline =     {
        "log_id": "test",
        "wandb_project": "vit-sae-test",
        "n_feats": 1024,
        "batch_size": 4096,
        "k": 128,
        "lr": 0.0004,
        "l1_coefficient":0.00008,
        'data_name': "ViT-3_000_000",
        "d_in": 1024,
        "samples_per_file": 20480,
        "seq_len": 1,
        "cache_buffer_size": 10,
        "n_cache_workers": 6,
        "outer_batch_size": 4096,
        "architecture": 'relu',
        "batch_norm": False,
    }

all_configs = []
for l1 in [0.0256, 0.0128, 0.0064,	0.0032,	0.0016,	0.0008,	0.0004]:
    for 
    config = baseline.copy()
    config['l1_coefficient'] = l1
    config['architecture'] = 'relu'
    all_configs.append(config)


for k in [8,	16,	32,	64,	128,	256]:

with open('cruft/relu_configs.json', 'w') as f:
    json.dump(all_configs, f)