from train_sae import main

configs = [
    {
        "name": "baseline-big-4x",
        "wandb_project": "vit-sae-test",
        "n_feats": 65536 * 4,
        "batch_size": 1024,
        "k": 32,
        "lr": 0.0004,
        "l1_coefficient":0.00008,
        'data_name': "ViT-3_000_000",
        "d_in": 1024,
        "samples_per_file": 20480,
        "seq_len": 1,
        "n_experts": 4,
        "cache_buffer_size": 10,
        "n_cache_workers": 6,
        "architecture": 'topk',
        "batch_norm": False,
    }
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)