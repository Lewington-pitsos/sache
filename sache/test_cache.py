from sache.cache import CloudWCache, BUCKET_NAME
import torch
import boto3
import pytest

def file_exists_on_aws(client, bucket_name, prefix):
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return 'Contents' in response

@pytest.fixture
def s3_client():
    prefix = 'test'
    client = boto3.client('s3')

    yield prefix, client

    response = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    for obj in response.get('Contents', []):
        client.delete_object(Bucket=BUCKET_NAME, Key=obj.get('Key'))
    
    exists_after_delete = file_exists_on_aws(client, BUCKET_NAME, prefix)
    assert not exists_after_delete

def test_cache(s3_client):
    test_prefix,s3 = s3_client

    exists_before = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    assert not exists_before

    c = CloudWCache(test_prefix, batch_size=1)
    activations = torch.rand(2, 16, 16) 
    id = c.append(activations)

    assert id is not None

    exists_after = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    print(f"File exists after appending: {exists_after}")
    assert exists_after

    loaded = c.load(id)
    assert torch.equal(activations, loaded)




