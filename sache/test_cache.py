import shutil
import os
from io import BytesIO
import json
from sache.cache import S3WCache, BUCKET_NAME, S3RCache, RBatchingCache, RCache
import torch
import boto3
import pytest

def file_exists_on_aws(client, bucket_name, prefix):
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return 'Contents' in response

@pytest.fixture
def s3_client():
    prefix = 'test'
    with open('.credentials.json') as f:
        credentials = json.load(f)
    
    client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

    yield prefix, client

    response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    for obj in response.get('Contents', []):
        client.delete_object(Bucket=BUCKET_NAME, Key=obj.get('Key'))
    
    exists_after_delete = file_exists_on_aws(client, BUCKET_NAME, prefix)
    assert not exists_after_delete


@pytest.fixture
def test_cache_dir():
    local_dir = 'data/testing'

    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    yield local_dir

    shutil.rmtree(local_dir)


def test_write_to_s3cache(s3_client):
    test_prefix, s3 = s3_client

    exists_before = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    assert not exists_before
    
    c = S3WCache(s3, test_prefix, save_every=1)
    activations = torch.rand(2, 16, 16) 
    id = c.append(activations)

    assert id is not None

    exists_after = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    print(f"File exists after appending: {exists_after}")
    assert exists_after

    loaded = c.load(id)
    assert loaded.shape == activations.shape
    assert loaded.dtype == activations.dtype
    assert torch.equal(activations, loaded)

def test_batched_cache(s3_client):
    s3_prefix, s3_client = s3_client

    metadata = {
        'batch_size': 32,
        'sequence_length': 16,
        'hidden_dim': 9,
        'batches_per_file': 1,
        'dtype': 'torch.float32',
        'shape': [32, 16, 9],
        'bytes_per_file': 32 * 16 * 9 * 4
    }

    s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + '/metadata.json', Body=json.dumps(metadata))

    inner_cache = S3RCache(s3_client, s3_prefix)
    cache = RBatchingCache(inner_cache, 2)
    
    activations = torch.rand(32, 16, 9)

    buffer = BytesIO()
    torch.save(activations, buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, BUCKET_NAME, s3_prefix + '/a.saved.pt')

    cache.sync()
    count = 0
    for _ in cache:
        count += 1
        pass

    assert count == 16

def test_s3_read_cache(s3_client):
    s3_prefix, s3_client = s3_client

    metadata = {
        'batch_size': 32,
        'sequence_length': 16,
        'hidden_dim': 9,
        'batches_per_file': 1,
        'dtype': 'torch.float32',
        'shape': [32, 16, 9],
        'bytes_per_file': 32 * 16 * 9 * 4
    }

    s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + '/metadata.json', Body=json.dumps(metadata))

    cache = S3RCache(s3_client, s3_prefix, concurrency=10)

    count = 0
    for batch in cache:
        count += 1
        pass

    assert count == 0

    activations = torch.rand(32, 16, 9)
    tensor_bytes = activations.numpy().tobytes()

    s3_client.put_object(
        Bucket=BUCKET_NAME, 
        Key=s3_prefix + '/a.saved.pt', 
        Body=tensor_bytes, 
        ContentLength=len(tensor_bytes),
        ContentType='application/octet-stream'
    )
    for batch in cache:
        count += 1
        pass

    assert count == 0

    cache.sync()
    for batch in cache:
        count += 1
        pass

    assert count > 0

    assert batch.shape == activations.shape
    assert batch.dtype == activations.dtype
    assert torch.equal(batch, activations)
    assert batch.isnan().sum().item() == 0
    cache.stop_downloading()

def test_local_reading_cache(test_cache_dir):
    cache = RCache(test_cache_dir, 'cpu', buffer_size=2, num_workers=1)

    count = 0
    for _ in cache:
        count += 1
        pass
    assert count == 0

    activations = torch.rand(32, 16, 9)
    torch.save(activations, os.path.join(test_cache_dir, 'a.pt'))

    for _ in cache:
        count += 1
        pass
    assert count == 0

    cache.sync()
    for batch in cache:
        count += 1
        pass
    assert count >= 0
    assert torch.equal(activations, batch)

def test_local_read_batching(test_cache_dir):
    inner_cache = RCache(test_cache_dir, 'cpu')
    cache = RBatchingCache(inner_cache, 2)

    activations = torch.rand(32, 16, 9)
    torch.save(activations, os.path.join(test_cache_dir, 'a.pt'))

    cache.sync()
    count = 0
    for _ in cache:
        count += 1
        pass

    assert count == 16