import shutil
import os
from io import BytesIO
import json
from sache.constants import BUCKET_NAME
from sache.cache import S3WCache, S3RCache, RBatchingCache, ShufflingRCache, MultiLayerS3WCache, metadata_path
import torch
import boto3
import pytest
import time
from unittest import mock


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
    
    c = S3WCache(s3, test_prefix, bucket_name=BUCKET_NAME, save_every=1)
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
        'd_in': 9,
        'batches_per_file': 1,
        'dtype': 'torch.float32',
        'shape': [32, 16, 9],
        'bytes_per_file': 32 * 16 * 9 * 4
    }

    s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + '/metadata.json', Body=json.dumps(metadata))

    inner_cache = S3RCache(s3_client, s3_prefix, BUCKET_NAME)
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
        'd_in': 9,
        'batches_per_file': 1,
        'dtype': 'torch.float32',
        'shape': [32, 16, 9],
        'bytes_per_file': 32 * 16 * 9 * 4
    }

    s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_prefix + '/metadata.json', Body=json.dumps(metadata))

    cache = S3RCache(s3_client, s3_prefix, BUCKET_NAME, concurrency=10)

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
    cache._stop_downloading()

    for batch in cache:
        count += 1
        pass
    assert count >= 0
    assert torch.equal(activations, batch)


class MockCache():
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        self.data = iter(self.data)
        return self

    def __next__(self):
        return next(self.data)

    def finalize(self):
        pass

    def sync(self):
        pass


def test_shuffling_read_cache():
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 8
    d_in = 16
    
    data = [torch.ones(batch_size, seq_len, d_in, dtype=torch.float32) * i for i in range(15)]
    mc = MockCache(data)

    sc = ShufflingRCache(mc, batch_size * seq_len * 6, batch_size * seq_len, d_in, dtype=torch.float32)

    for i, batch in enumerate(sc):
        assert batch.shape == (batch_size * seq_len, d_in)
        assert batch.isnan().sum().item() == 0
        assert batch.std().item() > 1.0

    assert i == 14


def test_shuffling_read_cache_uneven_batch_size():
    torch.manual_seed(42)

    batch_size = 5
    seq_len = 8
    d_in = 16
    
    data = [torch.ones(batch_size, seq_len, d_in, dtype=torch.float32) * i for i in range(15)]
    mc = MockCache(data)

    sc = ShufflingRCache(mc, seq_len * 29, batch_size * seq_len, d_in, dtype=torch.float32)

    for i, batch in enumerate(sc):
        assert batch.shape == (batch_size * seq_len, d_in)
        assert batch.isnan().sum().item() == 0
        assert batch.std().item() > 1.0

    assert i == 14


def test_small_bs_shuffling_read_cache():
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 8
    d_in = 16
    out_bs = 2
    
    data = [torch.ones(batch_size, seq_len, d_in, dtype=torch.float32) * i for i in range(15)]
    mc = MockCache(data)

    sc = ShufflingRCache(mc, batch_size * seq_len * 12, out_bs * seq_len, d_in, dtype=torch.float32)

    for i, batch in enumerate(sc):
        assert batch.shape == (out_bs * seq_len, d_in)
        assert batch.isnan().sum().item() == 0
        assert batch.std().item() > 1.0

    assert i == 29

def test_big_cache_bs_shuffling_read_cache():
    torch.manual_seed(42)

    batch_size = 6
    seq_len = 8
    d_in = 16
    out_bs = 2
    
    data = [torch.ones(batch_size, seq_len, d_in, dtype=torch.float32) * i for i in range(5)]
    mc = MockCache(data)

    sc = ShufflingRCache(mc, batch_size * seq_len * 12, out_bs * seq_len, d_in, dtype=torch.float32)

    for i, batch in enumerate(sc):
        assert batch.shape == (out_bs * seq_len, d_in)
        assert batch.isnan().sum().item() == 0
        assert batch.std().item() > 1.0

    assert i == 14

def test_multilayer_s3_wcache_activations(s3_client):
    test_prefix, s3 = s3_client

    # Read credentials
    with open('.credentials.json') as f:
        credentials = json.load(f)

    exists_before = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    assert not exists_before

    hook_locations = [(3, 'module1'), (7, 'module2')]
    cache = MultiLayerS3WCache(
        creds=credentials,
        hook_locations=hook_locations,
        run_name=test_prefix,
        max_queue_size=10,
        input_tensor_shape=(2, 16, 16),
        num_workers=2,
        bucket_name=BUCKET_NAME,
        dtype=torch.float32
    )

    activation_dict = {
        (3, 'module1'): torch.rand(2, 16, 16),
        (7, 'module2'): torch.rand(2, 16, 16)
    }

    cache.append(activation_dict)

    # Finalize to ensure uploads are complete
    cache.finalize()

    exists_after = file_exists_on_aws(s3, BUCKET_NAME, test_prefix)
    assert exists_after

    def get_location_name(run_name, location):
        layer, module = location
        return f'{run_name}/{layer}_{module}'

    # List the objects under the test_prefix
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=test_prefix)
    s3_objects = response.get('Contents', [])

    # Collect files by location
    files_by_location = {}
    for obj in s3_objects:
        key = obj['Key']
        if key.endswith('.saved.pt'):
            parts = key.split('/')
            if len(parts) >= 2:
                location_key = '/'.join(parts[:-1])  # Exclude the file name
                files_by_location.setdefault(location_key, []).append(key)

    for location, activations in activation_dict.items():
        loc_name = get_location_name(test_prefix, location)
        files = files_by_location.get(loc_name, [])
        assert len(files) == 1, f"Expected one file for location {loc_name}, got {len(files)}"

        key = files[0]
        # Download the object
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = response['Body'].read()
        # Reconstruct the tensor
        dtype = activations.dtype
        expected_size = activations.numel() * activations.element_size()
        assert len(data) == expected_size, f"Data size mismatch for {key}"

        loaded_tensor = torch.frombuffer(data, dtype=dtype).reshape(activations.shape)

        assert torch.equal(activations, loaded_tensor), f"Mismatch in activations for {key}"


def test_multilayer_s3_wcache_metadata(s3_client):
    test_prefix, s3 = s3_client

    # Define hook locations
    hook_locations = [('layer1', 'module1'), ('layer2', 'module2')]

    with open('.credentials.json') as f:
        credentials = json.load(f)

    # Initialize MultiLayerS3WCache
    cache = MultiLayerS3WCache(
        creds=credentials,
        hook_locations=hook_locations,
        run_name=test_prefix,
        max_queue_size=10,
        input_tensor_shape=(2, 16, 16),
        num_workers=2,
        bucket_name=BUCKET_NAME,
        dtype=torch.float32
    )

    # Define activation_dict with deterministic data for testing
    activation_dict = {
        ('layer1', 'module1'): torch.ones(2, 16, 16, dtype=torch.float32) * 3.14,
        ('layer2', 'module2'): torch.ones(2, 16, 16, dtype=torch.float32) * 2.71
    }

    # Append activations to the cache
    cache.append(activation_dict)

    # Compute expected metadata for each location
    expected_metadata = {}
    for location, activations in activation_dict.items():
        metadata = {
            'batch_size': activations.shape[0],
            'dtype': str(activations.dtype),
            'bytes_per_file': activations.element_size() * activations.numel(),
            'batches_per_file': 1,
            'shape': (activations.shape[0], *activations.shape[1:])
        }
        if len(activations.shape) == 3:
            metadata['sequence_length'] = activations.shape[1]
            metadata['d_in'] = activations.shape[2]
        elif len(activations.shape) == 2:
            metadata['d_in'] = activations.shape[1]
        else:
            raise ValueError(f"Unexpected activations shape: {activations.shape}")
        
        expected_metadata[location] = metadata

    cache.finalize()

    # Helper function to construct location name
    def get_location_name(run_name, location):
        layer, module = location
        return f'{run_name}/{layer}_{module}'

    # Verify that metadata.json exists and is correct for each location
    for location in hook_locations:
        loc_name = get_location_name(test_prefix, location)
        metadata_key = metadata_path(loc_name)

        # Check if metadata.json exists on S3
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=metadata_key)
        assert 'Contents' in response, f"Metadata file {metadata_key} does not exist on S3."
        assert any(obj['Key'] == metadata_key for obj in response['Contents']), f"Metadata file {metadata_key} not found in S3 contents."

        # Download metadata.json from S3
        response = s3.get_object(Bucket=BUCKET_NAME, Key=metadata_key)
        metadata_content = response['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_content)

        # Retrieve expected metadata
        expected = expected_metadata[location]

        # Compare metadata fields
        for key, value in expected.items():
            assert key in metadata, f"Metadata key '{key}' missing for location {loc_name}."
            if type(value) == tuple and type(metadata[key]) == list:
                value = list(value)
            assert metadata[key] == value, f"Metadata mismatch for key '{key}' in location {loc_name}."




def test_multi_layer_s3wcache_cleanup_on_exception():
    """
    Test that MultiLayerS3WCache properly terminates all child processes when an exception occurs.
    """
    creds = {'AWS_ACCESS_KEY_ID': 'dummy_key', 'AWS_SECRET': 'dummy_secret'}
    hook_locations = [('layer1', 'resid'), ('layer2', 'resid')]
    run_name = 'test_run'
    max_queue_size = 10
    input_tensor_shape = (1024, 768)  # Example shape
    num_workers = 2
    bucket_name = 'test_bucket'

    # Patch the worker_process to use dummy_worker instead for testing
        # Use pytest's raises to expect an exception
    try:
        with MultiLayerS3WCache(
            creds=creds,
            hook_locations=hook_locations,
            run_name=run_name,
            max_queue_size=max_queue_size,
            input_tensor_shape=input_tensor_shape,
            num_workers=num_workers,
            bucket_name=bucket_name
        ) as cache:
            # Simulate some work by putting items into the queue
            for i in range(max_queue_size):
                cache.to_upload.put((i, 'layer1'))

            # Allow some time for workers to process
            time.sleep(1)

            # Intentionally raise an exception to trigger cleanup
            raise ValueError("Intentional error for testing.")
    except ValueError:
        for worker in cache.workers:
            worker.join(timeout=1)  # Wait briefly for process to terminate
            assert not worker.is_alive(), f"Worker {worker.name} should have been terminated."