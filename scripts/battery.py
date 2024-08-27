# scale LR with batch size
# apply warmup (constant until training is kind of stable)
# break up each sample into N and calculate and apply batch norm over each subsample
# do the momentum things

import json
from train_sae import main

if __name__ == '__main__':
    with open('cruft/configs.json') as f:
        configs = json.load(f)

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)