import boto3
import torch
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3WCache

def main():
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    
    cache = S3WCache(s3_client, 'cache-upload-cruft', save_every=1)

    for i in range(1, 6):
        tensor = torch.randn(1536, 1024, 768)

        print('starting upload')
        id = cache.append(tensor)
        print(f"Appended tensor with id {id}")


if __name__ == "__main__":
    main()
