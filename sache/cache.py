import threading
import queue
import shutil
import os
import torch
from uuid import uuid4
import boto3
from io import BytesIO
import time

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


class ThreadedCache:
    def __init__(self, cache):
        self.cache = cache
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.start()

    def _worker(self):
        while True:
            method, args, kwargs = self.task_queue.get()
            if method is None:
                break
            acquired = self.lock.acquire(blocking=False)
            if acquired:
                try:
                    method(*args, **kwargs)
                finally:
                    self.lock.release()
            else:
                self.task_queue.put((method, args, kwargs))
            self.task_queue.task_done()
            time.sleep(0.01)

    def _run_in_thread(self, method, *args, **kwargs):
        self.task_queue.put((method, args, kwargs))

    def append(self, activations):
        self._run_in_thread(self.cache.append, activations)

    def finalize(self):
        self.close()
        self.cache.finalize()

    def close(self):
        self._run_in_thread(None)  # Signal the worker thread to exit
        self.worker_thread.join()


class S3WCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return S3WCache(s3_client, *args, **kwargs)

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


class S3RCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, s3_prefix, local_cache_dir, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return S3RCache(local_cache_dir, s3_client, s3_prefix, *args, **kwargs)

    def __init__(self, local_cache_dir, s3_client, s3_prefix, bucket_name=BUCKET_NAME) -> None:
        self.s3_prefix = s3_prefix
        self.s3_client = s3_client
        self.local_cache_dir = local_cache_dir
        self.bucket_name = bucket_name

        if not os.path.exists(self.local_cache_dir):
            os.makedirs(self.local_cache_dir, exist_ok=True)
        
        self._s3_paths = self._list_s3_files()

    def sync(self):
        self._s3_paths = self._list_s3_files()

    def _list_s3_files(self):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_prefix)
        return sorted([obj.get('Key') for obj in response.get('Contents', [])])

    def __iter__(self):
        self._file_index = 0
        return self

    def __next__(self):
        if self._file_index >= len(self._s3_paths):
            raise StopIteration

        s3_path = self._s3_paths[self._file_index]
        local_path = os.path.join(self.local_cache_dir, s3_path)

        if not os.path.exists(local_path):
            buffer = BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, s3_path, buffer)
            buffer.seek(0)
            activations = torch.load(buffer)

            parent_dir = os.path.dirname(local_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            torch.save(activations, local_path)
        else:
            activations = torch.load(local_path)

        self._file_index += 1
        return activations

    def clear_local(self):
        shutil.rmtree(self.local_cache_dir)

class S3RBatchingCache(S3RCache):
    def __init__(self, batch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.activations = None

        self._finished = False

    def __iter__(self):
        super().__iter__()
        self._finished = False
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration

        if self.activations is None:
            self.activations = super().__next__()

        while self.activations.shape[0] < self.batch_size:
            try:
                self.activations = torch.cat([self.activations, super().__next__()], dim=0)
            except StopIteration:
                self._finished = True
                if self.activations.shape[0] == 0:
                    raise StopIteration
                return self.activations

        batch = self.activations[:self.batch_size]

        self.activations = self.activations[self.batch_size:]

        return batch

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
