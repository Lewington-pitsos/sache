from train_sae import main

configs = [
    # { #     "log_id": "filterma_fix", #     "n_feats": 3072, #     "n_experts": 4, #     "batch_size": 65536, #     "shuffle": False, #     "lr": 0.0008, #     "wandb_project": "position_mse", #     "skip_first_n": 0, #     "filter_ma": True # }, # {
    #     "log_id": "no_filterma",
    #     "n_feats": 3072,
    #     "n_experts": 4,
    #     "batch_size": 65536,
    #     "shuffle": False,
    #     "lr": 0.0008,
    #     "wandb_project": "position_mse",
    #     "skip_first_n": 0,
    #     "filter_ma": False
    # },
    {
        "log_id": "gemma2",
        "n_feats": 3072,
        "n_experts": 4,
        "batch_size": 65536,
        "shuffle": False,
        "lr": 0.0008,
        "wandb_project": "position_mse",
        "skip_first_n": 0,
        "filter_ma": False,
        'run_name': "gemma2b",
        "d_in": 2048,
        "samples_per_file": 128,
        
    }
]

if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)