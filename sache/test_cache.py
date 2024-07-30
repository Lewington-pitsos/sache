import shutil
import os
import time
from io import BytesIO
import json
from sache.cache import S3WCache, BUCKET_NAME, S3RCache, S3RBatchingCache, RCache, INNER_CACHE_DIR
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

def test_cache(s3_client):
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
    assert torch.equal(activations, loaded)


def test_batched_cache(s3_client):
    s3_prefix, s3_client = s3_client
    cache = S3RBatchingCache(2, 'cache/test', s3_client, s3_prefix)
    
    activations = torch.rand(32, 16, 9)

    buffer = BytesIO()
    torch.save(activations, buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, BUCKET_NAME, s3_prefix + '/a.pt')

    cache.sync()
    count = 0
    for _ in cache:
        count += 1
        pass

    assert count == 16

def test_s3_read_cache(s3_client):
    s3_prefix, s3_client = s3_client
    cache = S3RCache('cache/test', s3_client, s3_prefix)

    count = 0
    for batch in cache:
        count += 1
        pass

    assert count == 0

    activations = torch.rand(32, 16, 9)

    buffer = BytesIO()
    torch.save(activations, buffer)
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, BUCKET_NAME, s3_prefix + '/a.pt')
    for batch in cache:
        count += 1
        pass

    assert count == 0

    cache.sync()
    for batch in cache:
        count += 1
        pass

    assert count > 0

    cache.clear_local()

@pytest.fixture
def test_cache_dir():
    local_dir = 'data/testing'

    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    yield local_dir

    shutil.rmtree(local_dir)


def test_local_reading_cache(test_cache_dir):
    cache = RCache(test_cache_dir, 'cpu')

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
    for _ in cache:
        count += 1
        pass
    assert count >= 0
