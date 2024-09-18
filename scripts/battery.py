from train_sae import main

configs = [
    {
        "log_id": "test",
        "wandb_project": "vit-sae-test",
        "n_feats": 1024 * 64,
        "batch_size": 1024,
        "k": 128,
        "lr": 0.0004,
        'data_name': "ViT-3_000_000",
        "d_in": 1024,
        "samples_per_file": 20480,
        "seq_len": 1,
        "cache_buffer_size": 10,
        "n_cache_workers": 6,
        "outer_batch_size": 4096,
        "architecture": 'relu'
    }
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)