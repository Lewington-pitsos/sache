import uuid
import torch
import os 
import sys
import json
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.constants import *

if __name__ == "__main__":
    with open('.credentials.json') as f:
        credentials = json.load(f)

    cache = S3RCache.from_credentials(
        aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], 
        aws_secret_access_key=credentials['AWS_SECRET'],  
        s3_prefix='merciless-citadel', 
        bucket_name='lewington-pitsos-sache', 
        chunk_size=MB * 16, 
        concurrency=200, 
        n_workers=4, 
        buffer_size=2
    )

    n = 250

    data_dir = 'data/250'

    for i, activations in tqdm.tqdm(enumerate(cache), total=n):
        if i == n:
            break

        id = str(uuid.uuid4())
        torch.save(activations[:, :, :64], data_dir + '/' + id + '.pt')

