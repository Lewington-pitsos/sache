from train_sae import main

configs = [
    {
        "log_id": "skip_0",
        "n_feats": 3072,
        "n_experts": 4,
        "batch_size": 63936,
        "shuffle": False,
        "lr": 0.0008,
        "wandb_project": "position_mse",
        "skip_first_n": 1
    },
    {
        "log_id": "skip_25",
        "n_feats": 3072,
        "n_experts": 4,
        "batch_size": 63936,
        "shuffle": False,
        "lr": 0.0008,
        "wandb_project": "position_mse",
        "skip_first_n": 25
    },
    {
        "log_id": "skip_100",
        "n_feats": 3072,
        "n_experts": 4,
        "batch_size": 63936,
        "shuffle": False,
        "lr": 0.0008,
        "wandb_project": "position_mse",
        "skip_first_n": 100
    }
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)