import boto3
import torch
import io
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import BUCKET_NAME

def generate_random_tensor(size_gb):
    """Generates a random tensor of the specified size in GB."""
    tensor_size = round((size_gb * 1024**3)) // 4  # float32 has 4 bytes
    tensor = torch.randn(tensor_size // (1024 * 1024), 1024 * 1024)
    return tensor

def upload_tensor_to_s3(bucket_name, tensor, s3_key):
    """Uploads a tensor directly to the specified S3 bucket."""
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    
    try:
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)
        print(f"Uploaded tensor to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading tensor: {e}")

def main():
    bucket_name = BUCKET_NAME

    for i in range(16):
        tensor = generate_random_tensor(5) 

        s3_key = f'tensors/tensor_{i}.pt'
        upload_tensor_to_s3(bucket_name, tensor, s3_key)

if __name__ == "__main__":
    main()
