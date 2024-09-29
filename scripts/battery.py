from train_sae import main
import json

# configs = [
# ]

filename = 'cruft/flop_switch_configs.json'
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