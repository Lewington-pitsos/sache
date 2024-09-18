from train_sae import main

configs = [
    {
        "log_id": "test",
        "wandb_project": "vit-sae-test",
        "n_feats": 4096,
        "batch_size": 4096 * 4,
        "shuffle": False,
        "lr": 0.0008,
        'data_name': "ViT-3_000_000",
        "d_in": 1024,
        "samples_per_file": 20480,
        "seq_len": 1,
        "cache_buffer_size": 6,
    }
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)