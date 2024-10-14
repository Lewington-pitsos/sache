import boto3
from moto import mock_aws
import pytest
from sache.train import get_checkpoint_from_s3

TEST_BUCKET_NAME = 'test-bucket'
TEST_CHECKPOINT_S3 = 's3://test-bucket/test_checkpoint.pt'
TEST_LOCAL_CHECKPOINT = 'test_checkpoint_local.pt'
MOCK_PREFIX = 'log/CLIP-ViT-L-14/17_resid/'

@pytest.fixture
def s3_setup():
    # Setup mock S3 using moto
    with mock_aws():
        s3 = boto3.client('s3')
        s3.create_bucket(Bucket=TEST_BUCKET_NAME)
        
        yield s3

def test_get_highest_checkpoint(s3_setup):
    # Add mock checkpoints to the S3 bucket
    s3_setup.put_object(Bucket=TEST_BUCKET_NAME, Key='log/CLIP-ViT-L-14/17_resid/a/100.pt', Body=b'')
    s3_setup.put_object(Bucket=TEST_BUCKET_NAME, Key='log/CLIP-ViT-L-14/17_resid/a/200.pt', Body=b'')
    s3_setup.put_object(Bucket=TEST_BUCKET_NAME, Key='log/CLIP-ViT-L-14/17_resid/b/150.pt', Body=b'')

    # Test that the function returns the highest checkpoint based on n_tokens.
    checkpoint, n_tokens = get_checkpoint_from_s3(s3_setup, TEST_BUCKET_NAME, MOCK_PREFIX)
    assert checkpoint == 's3://test-bucket/log/CLIP-ViT-L-14/17_resid/a/200.pt'
    assert n_tokens == 200

def test_reads_files_correctly(s3_setup):
    # Add a mock checkpoint to the S3 bucket
    s3_setup.put_object(Bucket=TEST_BUCKET_NAME, Key='log/CLIP-ViT-L-14/17_resid/a/100.pt', Body=b'')

    # Test that the function reads files correctly from the S3 bucket.
    checkpoint, n_tokens = get_checkpoint_from_s3(s3_setup, TEST_BUCKET_NAME, MOCK_PREFIX)
    assert checkpoint == 's3://test-bucket/log/CLIP-ViT-L-14/17_resid/a/100.pt'
    assert isinstance(n_tokens, int)
    assert n_tokens == 100

def test_no_checkpoints(s3_setup):
    # Test the case when there are no checkpoints in the S3 bucket.
    checkpoint, n_tokens = get_checkpoint_from_s3(s3_setup, TEST_BUCKET_NAME, MOCK_PREFIX)
    assert checkpoint is None
    assert n_tokens == 0
