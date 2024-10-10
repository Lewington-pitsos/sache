import json
import traceback  # Import the traceback module
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from sache import train_sae

def get_next_s3_key_from_sqs(sqs, queue_url):
    """Fetches the next S3 key from the SQS queue."""
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=30
        )
        
        messages = response.get('Messages', [])
        if not messages:
            print('No messages in SQS queue.')
            return None, None

        # Get the message and its receipt handle for deletion
        message = messages[0]
        receipt_handle = message['ReceiptHandle']
        config = message['Body']

        return config, receipt_handle
    except Exception as e:
        print(f'Error fetching S3 key from SQS: {e}')
        return None, None

def delete_message_from_sqs(sqs, queue_url, receipt_handle):
    try:
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
    except Exception as e:
        print(f'Error deleting message from SQS: {e}')

def keep_training():
    with open('.credentials.json') as f:
        credentials = json.load(f)

    sqs = boto3.client(
        'sqs',
        aws_access_key_id=credentials['AWS_ACCESS_KEY'],
        aws_secret_access_key=credentials['AWS_SECRET']
    )

    while True:
        config, receipt_handle = get_next_s3_key_from_sqs(sqs, credentials['SQS_TRAINING_CONFIG_QUEUE_URL'])
        if config is None:
            break

        try:
            print(f'Running with config: {config}')
            train_sae(credentials=credentials, **config)
            delete_message_from_sqs(sqs, credentials['SQS_TRAINING_CONFIG_QUEUE_URL'], receipt_handle)
        except Exception as e:
            print(f'Error running config {config}: {e}')
            traceback.print_exc()


        print('\nProceeding to the next config ----->\n')

