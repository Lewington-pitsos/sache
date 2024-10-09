import time
import json
import os
from uuid import uuid4
import boto3
import torch
import psutil
import wandb

LOG_DIR = 'log'

class SacheLogger():
    def __init__(self, run_name, s3_backup_bucket=None, s3_client=None, use_wandb=False, wandb_project=None, log_id=None, print_logs=False):
        self.run_name = run_name
        self.print_logs = print_logs

        if log_id is not None:
            self.log_id = log_id + '_' + str(uuid4())[:6]

            if os.path.exists(self._log_filename()):
                raise ValueError(f"Log file {self._log_filename()} already exists, please provide a unique log_id")
        else:
            self.log_id = str(uuid4())

        self.use_wandb = use_wandb

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)

        print('Logging to', self._log_filename())

        self.s3_backup_bucket = s3_backup_bucket
        if s3_backup_bucket is not None:
            assert s3_client is not None, 's3_client must be provided if s3_backup_bucket is provided'
            self.s3_backup_path = os.path.join(LOG_DIR, self.run_name, self.log_id + '.jsonl')
            self.s3_client = s3_client

        if use_wandb:
            if wandb_project is None:
                wandb_project = self.run_name
            
            self.wandb_project = wandb_project
            wandb.init(project=wandb_project, name=self.log_id)

    def _log_filename(self):
        return os.path.join(LOG_DIR,  self.run_name + '_' + self.log_id + '.jsonl')

    def _log_local(self, data):
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        if 'hr_timestamp' not in data:
            data['hr_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data['timestamp']))

        with open(self._log_filename(), 'a') as f:
            f.write(json.dumps(data) + '\n')

        if self.print_logs:
            print(data)

    def log_gpu_usage(self):
        if torch.cuda.is_available():
            self.log({
                'event': 'system_usage',
                'torch.cuda.memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,
                'torch.cuda.memory_reserved': torch.cuda.memory_reserved(0) / 1024**2,
                'torch.cuda.max_memory_reserved': torch.cuda.max_memory_reserved(0) / 1024**2,
                'cpu_percent': psutil.cpu_percent(),
            })

    def log_params(self, params):
        if self.use_wandb:
            wandb.config.update(params)

        self._log_local(params)

    def log(self, data):
        if self.use_wandb:
            wandb_message = {}

            for k, v in data.items():
                if isinstance(v, (int, float)):
                    wandb_message[k] = v

            wandb.log(wandb_message)

        self._log_local(data)

    def finalize(self):
        if self.s3_backup_bucket is not None: 
            self.s3_client.upload_file(self._log_filename(), self.s3_backup_bucket, self.s3_backup_path)
            print(f"Uploaded log file to {self.s3_backup_path}")

        if self.use_wandb:
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()
        if exc_type:
            print(f"An error occurred: {exc_value}")

def download_logs(bucket_name, s3_folder_prefix=LOG_DIR, local_download_path=LOG_DIR):
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
            local_file_path = os.path.join(local_download_path, file_path[len(s3_folder_prefix) + 1:].replace('/', '_'))
            
            # Create local directory structure if it does not exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3_client.download_file(bucket_name, file_path, local_file_path)
            print(f"Downloaded {file_path} to {local_file_path}")

class NOOPLogger:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None
