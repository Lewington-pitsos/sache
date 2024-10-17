import os
import boto3
import pytest
import json
from moto import mock_aws

from sache.log import SacheLogger

TEST_BUCKET_NAME = 'test-bucket'
TEST_RUN_NAME = 'test-run'

@pytest.fixture
def s3_setup():
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=TEST_BUCKET_NAME)
        yield s3

def test_sache_logger_upload(s3_setup, tmp_path):
    s3_client = s3_setup
    # Change current directory to tmp_path
    os.chdir(tmp_path)
    base_log_dir = 'log'  # Use 'log' as in the default LOG_DIR
    run_name = TEST_RUN_NAME
    s3_backup_bucket = TEST_BUCKET_NAME

    # Initialize the logger
    logger = SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,  # Disable wandb for testing
    )

    # Log some data
    logger.log({'event': 'test_event', 'value': 42})

    # Finalize to trigger the upload
    logger.finalize()

    # Get the s3_backup_path from the logger
    s3_backup_path = logger.s3_backup_path

    # Verify that the log file has been uploaded to S3
    response = s3_client.list_objects_v2(Bucket=s3_backup_bucket, Prefix=s3_backup_path)
    assert 'Contents' in response
    assert any(obj['Key'] == s3_backup_path for obj in response['Contents'])

    # Download the uploaded file and check its content
    downloaded_file = tmp_path / 'downloaded_log.jsonl'
    s3_client.download_file(s3_backup_bucket, s3_backup_path, str(downloaded_file))

    with open(downloaded_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['event'] == 'test_event'
        assert data['value'] == 42

def test_sache_logger_download(s3_setup, tmp_path):
    s3_client = s3_setup
    # Change current directory to tmp_path
    os.chdir(tmp_path)
    base_log_dir = 'log'  # Use 'log' as in the default LOG_DIR
    run_name = TEST_RUN_NAME
    s3_backup_bucket = TEST_BUCKET_NAME
    log_id = 'test-log-id'

    # Initialize the logger to get the s3_backup_path
    logger = SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,
        log_id=log_id,  # Use a specific log_id
    )

    logger.log({'event': 'existing_event', 'value': 100})

    # Finalize to trigger the upload
    logger.finalize()

    expected_log_file = logger._log_filename()
    assert os.path.exists(expected_log_file)
    os.remove(expected_log_file)
    assert not os.path.exists(expected_log_file)

    logger = SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,
        log_id=log_id,  # Use the same log_id
    )

    assert os.path.exists(expected_log_file)
    with open(expected_log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['event'] == 'existing_event'
        assert data['value'] == 100

def test_sache_logger_overwrite_s3(s3_setup, tmp_path):
    s3_client = s3_setup
    os.chdir(tmp_path)
    base_log_dir = 'log'
    run_name = TEST_RUN_NAME
    s3_backup_bucket = TEST_BUCKET_NAME
    log_id = 'test-log-id'

    # First logger instance
    logger1 = SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,
        log_id=log_id,
    )

    # Log initial data
    logger1.log({'event': 'first_event', 'value': 1})
    logger1.remote_sync()

    # Verify initial upload to S3
    s3_backup_path = logger1.s3_backup_path
    response = s3_client.list_objects_v2(Bucket=s3_backup_bucket, Prefix=s3_backup_path)
    assert 'Contents' in response
    assert any(obj['Key'] == s3_backup_path for obj in response['Contents'])
    downloaded_file = tmp_path / 'downloaded_log.jsonl'
    s3_client.download_file(s3_backup_bucket, s3_backup_path, str(downloaded_file))
    with open(downloaded_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['event'] == 'first_event'
        assert data['value'] == 1

    # Remove local log file to simulate a fresh start
    expected_log_file = logger1._log_filename()
    if os.path.exists(expected_log_file):
        os.remove(expected_log_file)
    assert not os.path.exists(expected_log_file)

    # Second logger instance with the same log_id
    logger2 = SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,
        log_id=log_id,
    )

    # Confirm the local log file is recreated (downloaded from S3)
    assert os.path.exists(expected_log_file)

    # Log new data
    logger2.log({'event': 'second_event', 'value': 2})
    logger2.remote_sync()

    # Download and verify content of the S3 file after overwrite
    s3_client.download_file(s3_backup_bucket, s3_backup_path, str(downloaded_file))
    with open(downloaded_file, 'r') as f:
        lines = f.readlines()
        # The new save should overwrite the old one, so we expect only one line
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data['event'] == 'first_event'
        assert data['value'] == 1

        data = json.loads(lines[1])
        assert data['event'] == 'second_event'
        assert data['value'] == 2

def test_sache_logger_context_manager(s3_setup, tmp_path):
    s3_client = s3_setup
    os.chdir(tmp_path)
    base_log_dir = 'log'
    run_name = TEST_RUN_NAME
    s3_backup_bucket = TEST_BUCKET_NAME
    log_id = 'context-manager-log-id'

    # Use the logger as a context manager
    with SacheLogger(
        run_name=run_name,
        base_log_dir=base_log_dir,
        s3_backup_bucket=s3_backup_bucket,
        s3_client=s3_client,
        use_wandb=False,
        log_id=log_id,
    ) as logger:
        # Log some data
        logger.log({'event': 'context_event', 'value': 123})

        # At this point, the log file should exist locally
        expected_log_file = logger._log_filename()
        assert os.path.exists(expected_log_file)

    # After exiting the context, finalize should have been called, and the log should be uploaded to S3
    s3_backup_path = logger.s3_backup_path
    response = s3_client.list_objects_v2(Bucket=s3_backup_bucket, Prefix=s3_backup_path)
    assert 'Contents' in response
    assert any(obj['Key'] == s3_backup_path for obj in response['Contents'])

    # Download and verify the content of the S3 file
    downloaded_file = tmp_path / 'downloaded_log_context.jsonl'
    s3_client.download_file(s3_backup_bucket, s3_backup_path, str(downloaded_file))
    with open(downloaded_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data['event'] == 'context_event'
        assert data['value'] == 123