from train_sae import main
import json

# configs = [
#     {
#         "name": "baseline-big-4x-fix",
#         "wandb_project": "vit-sae-test",
#         "n_feats": 65536,
#         "batch_size": 8192,
#         "k": 32,
#         "lr": 0.0004,
#         "l1_coefficient":0.00008,
#         'data_name': "ViT-3_000_000",
#         "d_in": 1024,
#         "samples_per_file": 20480,
#         "seq_len": 1,
#         "n_experts": 8,
#         "cache_buffer_size": 10,
#         "n_cache_workers": 6,
#         "architecture": 'topk',
#         "batch_norm": False,
#         "use_wandb": False
#     },
# ]

filename = 'cruft/switch_configs.json'
with open(filename) as f:
    configs = json.load(f)

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        try:
            print(f'Running with config: {config}')
            main(**config)
        except Exception as e:
            print(f'Error running config {config}: {e}\n\n proceeding to the next config ----->')
            continue