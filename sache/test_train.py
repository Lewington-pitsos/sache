import boto3
from moto import mock_aws
import pytest
import torch
from sache.model import SAE
from sache.train import get_checkpoint_from_s3, save_checkpoint, load_checkpoint

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


def test_save_and_load_sae(tmp_path):
    # Initialize model parameters
    n_features = 10
    d_in = 5
    device = 'cpu'  # Use 'cuda' if GPU is available
    sae = SAE(n_features=n_features, d_in=d_in, device=device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # Initialize dead_latents
    dead_latents = torch.zeros(n_features, device=device)

    # Set other variables
    token_count = 100
    log_id = 'test_log_id'
    n_iter = 100
    data_name = 'test_data'

    # Save checkpoint to a temporary directory
    save_checkpoint(
        sae=sae,
        optimizer=optimizer,
        token_count=token_count,
        dead_latents=dead_latents,
        log_id=log_id,
        n_iter=n_iter,
        data_name=data_name,
        base_dir=str(tmp_path),
        s3_client=None,
        bucket_name=None
    )

    # Load checkpoint from the saved file
    model_filename = tmp_path / data_name / log_id / f'{n_iter}.pt'
    checkpoint = torch.load(model_filename)

    # Reconstruct model and optimizer
    sae_loaded = SAE(n_features=n_features, d_in=d_in, device=device)
    optimizer_loaded = torch.optim.Adam(sae_loaded.parameters(), lr=0.001)

    sae_loaded.load_state_dict(checkpoint['model_state_dict'])
    optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])

    # Check that token_count and other variables match
    assert checkpoint['token_count'] == token_count
    assert checkpoint['log_id'] == log_id
    assert checkpoint['n_iter'] == n_iter

    # Check that dead_latents match
    assert torch.allclose(dead_latents, checkpoint['dead_latents'])

    # Check that model parameters match
    for param_original, param_loaded in zip(sae.parameters(), sae_loaded.parameters()):
        assert torch.allclose(param_original, param_loaded), "Model parameters do not match after loading"

    # Check that optimizer states match
    for state_original, state_loaded in zip(optimizer.state.values(), optimizer_loaded.state.values()):
        for k in state_original:
            v_original = state_original[k]
            v_loaded = state_loaded[k]
            if isinstance(v_original, torch.Tensor):
                assert torch.allclose(v_original, v_loaded), f"Optimizer state {k} does not match after loading"
            else:
                assert v_original == v_loaded, f"Optimizer state {k} does not match after loading"


def test_save_checkpoint_and_find(s3_setup):
    # Initialize model parameters
    n_features = 10
    d_in = 5
    device = 'cpu'
    sae = SAE(n_features=n_features, d_in=d_in, device=device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # Initialize dead_latents
    dead_latents = torch.zeros(n_features, device=device)

    # Set other variables
    token_count = 100
    log_id = 'test_log_id'
    n_iter = token_count
    data_name = 'test_data'
    base_dir = 'log'

    # Save checkpoint to mock S3
    save_checkpoint(
        sae=sae,
        optimizer=optimizer,
        token_count=token_count,
        dead_latents=dead_latents,
        log_id=log_id,
        n_iter=n_iter,
        data_name=data_name,
        base_dir=base_dir,
        s3_client=s3_setup,
        bucket_name=TEST_BUCKET_NAME
    )

    # Use get_checkpoint_from_s3 to find the checkpoint
    checkpoint, n_tokens = get_checkpoint_from_s3(s3_setup, TEST_BUCKET_NAME, MOCK_PREFIX)

    # Expected checkpoint path
    expected_checkpoint = f's3://{TEST_BUCKET_NAME}/{base_dir}/{data_name}/{log_id}/{n_iter}.pt'
    assert checkpoint == expected_checkpoint
    assert n_tokens == token_count


def test_find_highest_checkpoint(s3_setup):
    # Initialize model parameters
    n_features = 10
    d_in = 5
    device = 'cpu'
    sae = SAE(n_features=n_features, d_in=d_in, device=device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.001)

    # Initialize dead_latents
    dead_latents = torch.zeros(n_features, device=device)

    # Set other variables
    log_id = 'test_log_id'
    data_name = 'test_data'
    base_dir = 'log'

    # Save multiple checkpoints to mock S3
    token_counts = [100, 200, 150]
    for token_count in token_counts:
        n_iter = token_count
        save_checkpoint(
            sae=sae,
            optimizer=optimizer,
            token_count=token_count,
            dead_latents=dead_latents,
            log_id=log_id,
            n_iter=n_iter,
            data_name=data_name,
            base_dir=base_dir,
            s3_client=s3_setup,
            bucket_name=TEST_BUCKET_NAME
        )

    # Use get_checkpoint_from_s3 to find the checkpoint with the highest n_tokens
    prefix = f'{base_dir}/{data_name}/{log_id}/'
    checkpoint, n_tokens = get_checkpoint_from_s3(s3_setup, TEST_BUCKET_NAME, prefix)

    # Expected checkpoint is the one with the highest token_count
    max_token_count = max(token_counts)
    expected_checkpoint = f's3://{TEST_BUCKET_NAME}/{base_dir}/{data_name}/{log_id}/{max_token_count}.pt'
    assert checkpoint == expected_checkpoint
    assert n_tokens == max_token_count

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
