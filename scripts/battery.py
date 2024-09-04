from train_sae import main

configs = [
    {
        "log_id": "0_skip_retry",
        "n_feats": 3072,
        "n_experts": 4,
        "batch_size": 65536,
        "shuffle": False,
        "lr": 0.0008,
        "wandb_project": "position_mse",
        "skip_first_n": 0,
        "filter_ma": True
    },
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)