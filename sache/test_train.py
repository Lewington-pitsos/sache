import os
import boto3
import torch
from unittest.mock import MagicMock, patch
from moto import mock_aws

from sache.train import load_sae_from_checkpoint 

TEST_BUCKET_NAME = 'test-bucket'
TEST_CHECKPOINT_S3 = 's3://test-bucket/test_checkpoint.pt'
TEST_LOCAL_CHECKPOINT = 'test_checkpoint_local.pt'

@mock_aws
def test_load_sae_from_checkpoint_s3():
    # Create a mocked S3 client and bucket
    s3_client = boto3.client('s3', region_name='us-east-1')
    s3_client.create_bucket(Bucket=TEST_BUCKET_NAME)

    # Mock tensor to save and upload
    tensor = torch.tensor([1.0, 2.0, 3.0])
    local_file_path = 'test_checkpoint.pt'
    torch.save(tensor, local_file_path)

    # Upload the tensor to the mocked S3 bucket
    with open(local_file_path, 'rb') as f:
        s3_client.put_object(Bucket=TEST_BUCKET_NAME, Key='test_checkpoint.pt', Body=f)

    # Mock the local directory and function call
    with patch('sache.train.os.makedirs') as mock_makedirs:
        with patch('sache.train.os.path.exists', return_value=False):
            result_tensor = load_sae_from_checkpoint(TEST_CHECKPOINT_S3, s3_client, local_dir='cruft')

            # Verify that the local directory was created
            mock_makedirs.assert_called_once_with('cruft')
            # Verify that the tensor was loaded correctly
            assert torch.equal(result_tensor, tensor)

    # Clean up the local file
    os.remove(local_file_path)

@patch('sache.train.os.path.exists', return_value=True)
def test_load_sae_from_checkpoint_local(mock_exists):
    # Mock tensor to save locally
    tensor = torch.tensor([4.0, 5.0, 6.0])
    torch.save(tensor, TEST_LOCAL_CHECKPOINT)

    # Call the function with the local path
    result_tensor = load_sae_from_checkpoint(TEST_LOCAL_CHECKPOINT, MagicMock())

    # Verify that the tensor was loaded correctly
    assert torch.equal(result_tensor, tensor)

    # Clean up the local file
    os.remove(TEST_LOCAL_CHECKPOINT)
