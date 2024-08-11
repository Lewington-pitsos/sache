import time
import json
import os
from uuid import uuid4
import boto3

from sache.constants import BUCKET_NAME

LOG_DIR = 'log'

class ProcessLogger():
    def __init__(self, run_name, s3_backup_bucket=None, s3_client=None):
        self.run_name = run_name
        self.logger_id = str(uuid4())

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)

        print('Logging to', self._log_filename())

        self.s3_backup_bucket = s3_backup_bucket
        if s3_backup_bucket is not None:
            assert s3_client is not None, 's3_client must be provided if s3_backup_bucket is provided'
            self.s3_backup_path = os.path.join(LOG_DIR, self.run_name, self.logger_id + '.jsonl')
            self.s3_client = s3_client

    def _log_filename(self):
        return os.path.join(LOG_DIR,  self.run_name + '_' + self.logger_id + '.jsonl')

    def log(self, data):
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        if 'hr_timestamp' not in data:
            data['hr_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data['timestamp']))

        with open(self._log_filename(), 'a') as f:
            f.write(json.dumps(data) + '\n')

    def finalize(self):
        if self.s3_backup_bucket is not None: 
            self.s3_client.upload_file(self._log_filename(), self.s3_backup_bucket, self.s3_backup_path)
            print(f"Uploaded log file to {self.s3_backup_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()
        if exc_type:
            print(f"An error occurred: {exc_value}")

def download_logs(bucket_name=BUCKET_NAME, s3_folder_prefix=LOG_DIR, local_download_path=LOG_DIR):
    # Load credentials
    with open('.credentials.json') as f:
        credentials = json.load(f)

    # Create an S3 client
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

    # List objects in the specified folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder_prefix)

    # Check if there are contents
    if 'Contents' in response:
        for item in response['Contents']:
            file_path = item['Key']
            local_file_path = os.path.join(local_download_path, file_path[len(s3_folder_prefix):])

            # Create local directory structure if it does not exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3_client.download_file(bucket_name, file_path, local_file_path)
            print(f"Downloaded {file_path} to {local_file_path}")