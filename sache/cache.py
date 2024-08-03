import json
import threading
import queue
import os
import torch
from uuid import uuid4
import boto3
from io import BytesIO
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import asyncio
import aiohttp

STAGES = ['saved', 'shuffled']
OUTER_CACHE_DIR = 'cache'
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


def _metadata_path(run_name):
    return f'{run_name}/metadata.json'

class S3WCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return S3WCache(s3_client, *args, **kwargs)

    def __init__(self, s3_client, run_name, save_every=1):
        self.save_every = save_every
        self.run_name = run_name
        self._in_mem = []
        self.s3_client = s3_client
        self.metadata = None

    def append(self, activations):
        if self.metadata is None:
            self.metadata = {
                'batch_size': activations.shape[0],
                'sequence_length': activations.shape[1],
                'dtype': str(activations.dtype),
                'hidden_dim': activations.shape[2],
                'bytes_per_file': activations.element_size() * activations.numel(),
                'batches_per_file': self.save_every,
                'shape': (activations.shape[0] * self.save_every, *activations.shape[1:])
            }

            self.s3_client.put_object(Body=json.dumps(self.metadata), Bucket=BUCKET_NAME, Key=_metadata_path(self.run_name))
        else:
            if activations.shape[0] != self.metadata['batch_size']:
                print(f'Warning: batch size mismatch. Expected {self.metadata["batch_size"]}, got {activations.shape}')
            if activations.shape[1] != self.metadata['sequence_length']:
                print(f'Warning: sequence length mismatch. Expected {self.metadata["sequence_length"]}, got {activations.shape}')
            if activations.shape[2] != self.metadata['hidden_dim']:
                print(f'Warning: hidden dim mismatch. Expected {self.metadata["hidden_dim"]}, got {activations.shape}')
            if str(activations.dtype) != self.metadata['dtype']:
                print(f'Warning: dtype mismatch. Expected {self.metadata["dtype"]}, got {activations.dtype}')

        self._in_mem.append(activations)

        if len(self._in_mem) == self.save_every:
            return self._save_in_mem()

        return None

    def finalize(self):
        # effectively a no-op since append will always save unless there are too few activations
        # to make up the block, in which case we do not want to save them or we lose file size uniformity
        if self.metadata is None:
            raise ValueError('Cannot finalize cache without any data')

    def _save_in_mem(self):
        id = self._save(torch.cat(self._in_mem), str(uuid4()))
        self._in_mem = []

        return id

    def _filename(self, id):
        return f'{self.run_name}/{id}.saved.pt'

    def _save(self, activations, id):
        filename = self._filename(id)

        tensor_bytes = activations.numpy().tobytes()
    
        self.s3_client.put_object(
            Bucket=BUCKET_NAME, 
            Key=filename, 
            Body=tensor_bytes, 
            ContentLength=len(tensor_bytes),
            ContentType='application/octet-stream'
        )

        return id

    def load(self, id):
        filename = self._filename(id)
        buffer = BytesIO()
        self.s3_client.download_fileobj(BUCKET_NAME, filename, buffer)
        buffer.seek(0)

        return torch.frombuffer(buffer.read(), dtype=torch.float32).reshape(self.metadata['shape'])

class RCache():
    def __init__(self, local_cache_dir, device, buffer_size=10, num_workers=4):
        self.cache_dir = local_cache_dir
        self.device = device
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        self.executor = None
        self.buffer_filler_thread = None
        self.stop_filling = False
        self.sync()
    
    def sync(self):
        self.stop_filling = True
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        if self.buffer_filler_thread is not None:
            self.buffer_filler_thread.join()

        self.files = os.listdir(self.cache_dir)
        self._file_idx = 0
        self.buffer = Queue(maxsize=self.buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.stop_filling = False
        self.buffer_filler_thread = Thread(target=self._fill_buffer)
        self.buffer_filler_thread.start()

    def _fill_buffer(self):
        while self._file_idx < len(self.files) and not self.stop_filling:
            if self.buffer.full():
                time.sleep(0.1)
            else:
                self.executor.submit(self._load_file, self._file_idx)
                self._file_idx += 1

    def _load_file(self, idx):
        if idx < len(self.files):
            filename = os.path.join(self.cache_dir, self.files[idx])
            activations = torch.load(filename, weights_only=True, map_location=self.device)
            self.buffer.put(activations)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer.empty() and self._file_idx >= len(self.files):
            if self.executor is not None:
                self.executor.shutdown(wait=True) 
                self.executor = None
            self.stop_filling = True
            if self.buffer_filler_thread is not None:
                self.buffer_filler_thread.join()
                self.buffer_filler_thread = None
            raise StopIteration

        return self.buffer.get(block=True)
    
    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        self.stop_filling = True
        if self.buffer_filler_thread is not None:
            self.buffer_filler_thread.join()

KB = 1024
MB = KB * KB


async def download_chunks(session, url, total_size, chunk_size):
    chunks = [(i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)]

    tasks = [asyncio.create_task(request_chunk(session, url, start, end)) for start, end in chunks]
    return await asyncio.gather(*tasks)

async def request_chunk(session, url, start, end):
    headers = {
        "Range": f"bytes={start}-{end}",
    }
    async with session.get(url, headers=headers) as response:
        return start, await response.read()

class S3RCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, s3_prefix, local_cache_dir, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return S3RCache(local_cache_dir, s3_client, s3_prefix, *args, **kwargs)

    def __init__(self, s3_client, s3_prefix, bucket_name=BUCKET_NAME, device='cpu', concurrency=100, chunk_size=MB*16) -> None:
        self.s3_prefix = s3_prefix
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.device = device
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        
        self._s3_paths = self._list_s3_files()

        response = self.s3_client.get_object(Bucket=bucket_name, Key=_metadata_path(s3_prefix))
        content = response['Body'].read()
        self.metadata = json.loads(content)

        self.downloading_thread = None
        self.buffer = []

    def sync(self):
        self._s3_paths = self._list_s3_files()

    def _list_s3_files(self):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_prefix)
        _metadata = _metadata_path(self.s3_prefix)
        paths = [f"http://{BUCKET_NAME}.s3.amazonaws.com/{obj['Key']}" for obj in response['Contents'] if obj['Key'] != _metadata]

        return sorted(paths)

    def __iter__(self):
        self._file_index = 0

        if self.downloading_thread is not None:
            raise ValueError('Cannot iterate over cache a second time while it is downloading')

        if len(self._s3_paths) > 0:
            self.downloading_thread = Thread(target=self._download_loop)
            self.downloading_thread.start()

        return self

    def _download_loop(self):
        asyncio.run(self._async_download())

    async def _async_download(self):   
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._file_index < len(self._s3_paths):
                if len(self.buffer) < 3:
                    url = self._s3_paths[self._file_index]
                    results = await download_chunks(session, url, self.metadata['bytes_per_file'], self.chunk_size)
                    self.buffer.append(results)
                    self._file_index += 1
                else:
                    await asyncio.sleep(0.1)


    def __next__(self):
        while self._file_index < len(self._s3_paths) or self.buffer:
            if self.buffer:
                buffers = self.buffer.pop(0)

                combined_bytes = b''.join(chunk for _, chunk in sorted(buffers, key=lambda x: x[0])) 

                t = torch.frombuffer(combined_bytes, dtype=eval(self.metadata['dtype']))
                return t.reshape(self.metadata['shape'])
            else:
                time.sleep(0.1)

        if self.downloading_thread is not None:
            self.stop_downloading()()
        raise StopIteration

    def stop_downloading(self):
        if self.downloading_thread is not None:
            self.downloading_thread.join()
            self.downloading_thread = None

class RBatchingCache():
    def __init__(self, cache, batch_size) -> None:
        self.cache = cache
        self.batch_size = batch_size
        self.activations = None

        self._finished = False

    def sync(self):
        self.cache.sync()

    def __iter__(self):
        self._finished = False
        self.cache.__iter__()
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration

        if self.activations is None:
            self.activations = next(self.cache)

        while self.activations.shape[0] < self.batch_size:
            try:
                self.activations = torch.cat([self.activations, next(self.cache)], dim=0)
            except StopIteration:
                self._finished = True
                if self.activations.shape[0] == 0:
                    raise StopIteration
                return self.activations

        batch = self.activations[:self.batch_size]

        self.activations = self.activations[self.batch_size:]

        return batch

class WCache():
    def __init__(self, run_name, save_every=1, base_dir=OUTER_CACHE_DIR):
        self.save_every = save_every

        self.outer_cache_dir = os.path.join(base_dir, run_name)
        self.cache_dir = os.path.join(self.outer_cache_dir, INNER_CACHE_DIR)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self._in_mem = []

    def n_saved(self):
        return len(os.listdir(self.cache_dir))

    def append(self, activations):
        self._in_mem.append(activations)

        if len(self._in_mem) >= self.save_every:
            self._save_in_mem()

    def finalize(self):
        if self._in_mem:
            self._save_in_mem()

    def _save_in_mem(self):
        self._save(torch.cat(self._in_mem), 'saved', str(uuid4()))
        self._in_mem = []

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
