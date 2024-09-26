from train_sae import main
import json

configs = [
    {
    "wandb_project": "vit-switch",
    "n_feats": 65536,
    "n_tokens": 3000000,
    "batch_size": 1024,
    "k": 8,
    "lr": 0.0004,
    "l1_coefficient": 1e-04,
    "data_name": "ViT-3mil",
    "d_in": 1024,
    "seq_len": 1,
    "cache_buffer_size": 3,
    "n_cache_workers": 4,
    "batch_norm": False,
    "save_every": 4000000,
    "architecture": "relu",
    "name": "relu-l1-0.0001"
  },
    {
    "wandb_project": "vit-switch",
    "n_feats": 65536,
    "n_tokens": 3000000,
    "batch_size": 1024,
    "k": 8,
    "lr": 0.0004,
    "l1_coefficient": 0.00011,
    "data_name": "ViT-3mil",
    "d_in": 1024,
    "seq_len": 1,
    "cache_buffer_size": 3,
    "n_cache_workers": 4,
    "batch_norm": False,
    "save_every": 4000000,
    "architecture": "relu",
    "name": "relu-l1-0.00011"
  },
]

# filename = 'cruft/switch_configs.json'
# with open(filename) as f:
#     configs = json.load(f)

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        try:
            print(f'Running with config: {config}')
            main(**config)
        except Exception as e:
            print(f'Error running config {config}: {e}\n\n proceeding to the next config ----->')
            continue