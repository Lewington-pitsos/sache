import os
import torch
from multiprocessing import Process
import randomname

from sache.cache import Cache
from sache.generator import generate
from sache.shuffler import shuffle
from sache.trainer import train

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


params = {
    'epochs': 10,
    'transformer_name': 'gpt2',
    'dataset_path': 'NeelNanda/pile-10k'
}

run_name = randomname.generate('adj/', 'n/', 'n/')
base_cache_dir = 'cache'
cache_dir = os.path.join(base_cache_dir, run_name)

cache = Cache(cache_dir=cache_dir)

generate_process = Process(target=generate, kwargs=({'cache_dir': cache_dir, **params}))
generate_process.start()

shuffle_process = Process(target=shuffle, kwargs=({'cache_dir': cache_dir}))
shuffle_process.start()

train(cache_dir, **params)

generate_process.join()
shuffle_process.join()


