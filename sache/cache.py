import random
import os
import torch
from uuid import uuid4
import boto3
from io import BytesIO

STAGES = ['saved', 'shuffled']
BASE_CACHE_DIR = 'cache'
INNER_CACHE_DIR = 'cache'
BUCKET_NAME = 'lewington-pitsos-sache'

class NoopCache():
    def __init__(self, *args, **kwargs):
        pass

    def append(self, activations):
        pass

    def finalize(self):
        pass

class CloudWCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return CloudWCache(s3_client, *args, **kwargs)

    def __init__(self, s3_client, run_name, save_every=1):
        self.batch_size = save_every
        self.run_name = run_name
        self._in_mem = []
        self.s3_client = s3_client

    def append(self, activations):
        self._in_mem.append(activations)

        if len(self._in_mem) >= self.batch_size:
            return self._save_in_mem()

        return None

    def finalize(self):
        if self._in_mem:
            self._save_in_mem()

    def _save_in_mem(self):
        id = self._save(torch.cat(self._in_mem), str(uuid4()))
        self._in_mem = []

        return id

    def _filename(self, id):
        return f'{self.run_name}/{id}.saved.pt'

    def _save(self, activations, id):
        filename = self._filename(id)

        buffer = BytesIO()
        torch.save(activations, buffer)
        buffer.seek(0)
        
        self.s3_client.upload_fileobj(buffer, BUCKET_NAME, filename)

        return id

    def load(self, id):
        filename = self._filename(id)
        buffer = BytesIO()
        self.s3_client.download_fileobj(BUCKET_NAME, filename, buffer)
        buffer.seek(0)

        return torch.load(buffer)

class WCache():
    def __init__(self, run_name, save_every=1, base_dir=BASE_CACHE_DIR):
        self.batch_size = save_every

        self.inner_cache_dir = os.path.join(base_dir, run_name)
        self.cache_dir = os.path.join(base_dir, run_name, INNER_CACHE_DIR)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self._in_mem = []
    

    def n_saved(self):
        return len(os.listdir(self.cache_dir))

    def _save_in_mem(self):
        self._save(torch.cat(self._in_mem), 'saved', str(uuid4()))
        self._in_mem = []

    def append(self, activations):
        self._in_mem.append(activations)

        if len(self._in_mem) >= self.batch_size:
            self._save_in_mem()

    def finalize(self):
        if self._in_mem:
            self._save_in_mem()

    def _filename(self, id, stage):
        return os.path.join(self.cache_dir, f'{id}.{stage}.pt')

    def _save(self, activations, stage, id):
        filename = self._filename(id, 'saving')
        torch.save(activations, filename)

        self._move_stage(id, 'saving', stage)

    def _move_stage(self, id, current_stage, next_stage):
        old_filename = self._filename(id, current_stage)
        new_filename = self._filename(id, next_stage)
        os.rename(old_filename, new_filename)

class WRCache(WCache):
    def __init__(self, run_name, read_stage=None, device=None):
        super().__init__(run_name)
        
        self._relevant = {}
        self.loan_stage = read_stage
        self.device = device

        if read_stage not in STAGES:
            raise ValueError(f'loan_stage must be among {STAGES} got {read_stage}')

    def __iter__(self):
        self.reset_buffer(self.loan_stage)
        for id, v in self._relevant.items():
            yield id, torch.load(v, map_location=self.device)

    def _load(self, id):
        return torch.load(self._relevant[id], map_location=self.device)

    def give_back(self, id, activations):
        self._save(activations, self.loan_stage, id)

    def take(self, id):
        activations = self._load(id)

        del self._relevant[id]

        return activations

    def take_random(self):
        ids = list(self._relevant.keys())
        id = random.choice(ids)

        return id, self._load(id)
        
    def reset_buffer(self):
        all_files = os.listdir(self.cache_dir)
        self._n_files = len(all_files)

        self._relevant = {}
        for file in all_files:
            id = file.split('.')[0]
            if file.endswith(f'.{self.loan_stage}.pt'):
                self._relevant[id] = os.path.join(self.cache_dir, file)


