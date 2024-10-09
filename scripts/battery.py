import json
import traceback  # Import the traceback module
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache import train_sae

filename = 'cruft/test_configs.json'
with open(filename) as f:
    configs = json.load(f)

with open('.credentials.json') as f:
    credentials = json.load(f)

if __name__ == '__main__':
    print(f'Running {len(configs)} configs')

    for config in configs:
        try:
            print(f'Running with config: {config}')
            train_sae(credentials=credentials, **config)
        except Exception as e:
            print(f'Error running config {config}: {e}')
            traceback.print_exc()  # Print the full traceback
            print('\nProceeding to the next config ----->\n')
            continue
