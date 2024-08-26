import boto3
import torch
import json
import time
import sys 
import os
import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache, ShufflingRCache, RBatchingCache
from sache.train import TrainLogger
from sache.model import SwitchSAE, TopKSwitchSAE, TopKSAE
from sache.constants import MB, BUCKET_NAME

def main(
        run_name = 'merciless-citadel',
        n_steps = 289, # 647 is the total, 288 means just over 300,000,000 tokens
        k = 32,
        n_feats = 24576,
        d_in = 768,
        batch_size = 4096,
        outer_batch_size = 8192 * 32,
        n_experts = 32,
        l1_coefficient = 2e-3,
        privilege_weighting = 2e-1,
        learning_rate = 3e-4,
        samples_per_file = 1024,
        tokens_till_latent_dies = 10_000_000,
        device = 'cuda',
        use_wandb=True,
        log_bucket=BUCKET_NAME,
        data_bucket=BUCKET_NAME,
        shuffle=True,
        wandb_project=None,
        base_expert=False,
        switch_sae=False,
    ):
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    cache = S3RCache(s3_client, run_name, data_bucket, chunk_size=MB * 16, concurrency=200, n_workers=4, buffer_size=3)
    total_size = cache.metadata['bytes_per_file']
    tokens_per_file = samples_per_file * 1024

    overall_start = time.time()
    start = time.time()
        
    token_count = 0
    for j, t in enumerate(cache):
        
        pass
        
    overall_end = time.time()
    print(f"Overall time taken: {overall_end - overall_start:.2f} seconds")
    print(f"Overall MB per second: {round(total_size / MB * n_steps) / (overall_end - overall_start):.2f}")

if __name__ == "__main__":
    fire.Fire(main)
