# scale LR with batch size
# apply warmup (constant until training is kind of stable)
# break up each sample into N and calculate and apply batch norm over each subsample
# do the momentum things

import json
from train_sae import main


filename = 'cruft/k3.json'
if __name__ == '__main__':
    with open(filename) as f:
        configs = json.load(f)

    print(f'running {len(configs)} configs')

    for config in configs:
        print(f'Running with config: {config}')
        main(**config)