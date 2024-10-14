import time
import json
import os
import trace
from uuid import uuid4
import boto3
import torch
import psutil
import wandb

LOG_DIR = 'log'

class SacheLogger():
    def __init__(
            self, 
            run_name,
            base_log_dir=LOG_DIR, 
            s3_backup_bucket=None, 
            s3_client=None, 
            use_wandb=False, 
            wandb_project=None, 
            log_id=None, 
            print_logs=False,
            credentials=None,
        ):
        self.run_name = run_name
        self.print_logs = print_logs
        self.log_dir = os.path.join(base_log_dir, self.run_name)

        if log_id is not None:
            self.log_id = log_id + '_' + str(uuid4())[:10]

            if os.path.exists(self._log_filename()):
                raise ValueError(f"Log file {self._log_filename()} already exists, please provide a unique log_id")
        else:
            self.log_id = str(uuid4())

        self.use_wandb = use_wandb

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        print('Logging to', self._log_filename())

        self.s3_backup_bucket = s3_backup_bucket
        if s3_backup_bucket is not None:
            assert s3_client is not None, 's3_client must be provided if s3_backup_bucket is provided'
            self.s3_backup_path = os.path.join(self.log_dir, self.log_id + '.jsonl')
            self.s3_client = s3_client

            response = s3_client.list_objects_v2(Bucket=s3_backup_bucket, Prefix=self.s3_backup_path)
            if 'Contents' in response:
                print('Log file already exists in S3, downloading')
                download_logs(s3_backup_bucket, s3_folder_prefix=self.log_dir, local_download_path=self._log_filename())

        if use_wandb:
            if credentials is not None:
                os.environ["WANDB_API_KEY"] = credentials['WANDB_API_KEY']

            if wandb_project is None:
                wandb_project = self.run_name
            
            self.wandb_project = wandb_project
            wandb.init(project=wandb_project, name=self.log_id, id=self.log_id, resume='must')

    def _log_filename(self):
        return os.path.join(self.log_dir, self.log_id + '.jsonl')

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
