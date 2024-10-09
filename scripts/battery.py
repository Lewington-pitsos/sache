from train_sae import train
import json
import traceback  # Import the traceback module

filename = 'cruft/test_configs.json'
with open(filename) as f:
    configs = json.load(f)

if __name__ == '__main__':
    print(f'Running {len(configs)} configs')

    for config in configs:
        try:
            print(f'Running with config: {config}')
            train(**config)
        except Exception as e:
            print(f'Error running config {config}: {e}')
            traceback.print_exc()  # Print the full traceback
            print('\nProceeding to the next config ----->\n')
            continue
