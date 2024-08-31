# scale LR with batch size
# apply warmup (constant until training is kind of stable)
# break up each sample into N and calculate and apply batch norm over each subsample
# do the momentum things

import json
from train_sae import main


configs = [

    # {
    #     "log_id": "[<bos>, t]",
    #     "run_name": "relaxed-cyclist",
    #     "n_feats": 16384,
    #     "n_experts": 32,
    #     "batch_size": 65536,
    #     "lr": 0.0008,
    #     "secondary_input": "[<bos>, t]",
    #     "wandb_project": "tokenized-sae",
    #     "shuffle": False
    # }

# {
#     "log_id": "remove_pos_1",
#     "n_feats": 3072,
#     "n_experts": 4,
#     "batch_size": 65472,
#     "shuffle": False,
#     "lr": 0.0008,
#     "wandb_project": "position_mse"
# }

{
    "log_id": "remove_pos_25",
    "n_feats": 3072,
    "n_experts": 4,
    "batch_size": 63936,
    "shuffle": False,
    "lr": 0.0008,
    "wandb_project": "position_mse"
}
# {
#     "log_id": "remove_pos_1",
#     "n_feats": 3072,
#     "n_experts": 4,
#     "batch_size": 65536,
#     "shuffle": False,
#     "lr": 0.0008,
#     "wandb_project": "position_mse"
# }


]



if __name__ == '__main__':
    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)