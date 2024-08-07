import warnings
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
from multiprocessing import Value, Process, Queue
import multiprocessing as mp
from sache.constants import *

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


class ThreadedReadCache:
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
                'bytes_per_file': activations.element_size() * activations.numel() * self.save_every,
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.frombuffer(buffer.read(), dtype=torch.float32)

        return t.reshape(self.metadata['shape'])


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

async def download_chunks(session, url, total_size, chunk_size):
    chunks = [(i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)]
    tasks = [asyncio.create_task(request_chunk(session, url, start, end)) for start, end in chunks]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for response in responses:
        if isinstance(response, Exception):
            # Handle the error (e.g., log it, retry, or raise it)
            print("Error occurred:", response)
        else:
            results.append(response)

    return results

async def request_chunk(session, url, start, end):
    headers = {"Range": f"bytes={start}-{end}"}
    async with session.get(url, headers=headers) as response:
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx, etc.)
        return start, await response.read()

def download_loop(*args):
    asyncio.run(_async_download(*args,))
        
def compile(byte_buffers, dtype, shape):
    combined_bytes = b''.join(chunk for _, chunk in sorted(byte_buffers, key=lambda x: x[0])) 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = torch.frombuffer(combined_bytes, dtype=dtype)
        t = t.clone()
    t = t.reshape(shape)

    return t

def write_tensor(t, buffer, writeable_tensors, readable_tensors, ongoing_downloads):
    idx = writeable_tensors.get(block=True)
    buffer[idx, :] = t
    readable_tensors.put(idx, block=True)

    with ongoing_downloads.get_lock():
        ongoing_downloads.value -= 1

async def _async_download(        
        buffer, 
        file_index, 
        s3_paths, 
        stop, 
        shape, 
        activation_dtype, 
        readable_tensors, 
        writeable_tensors, 
        ongoing_downloads,
        concurrency, 
        bytes_per_file,
        chunk_size
    ):   

    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        while file_index.value < len(s3_paths) and not stop.value:
            with ongoing_downloads.get_lock():
                ongoing_downloads.value += 1

            with file_index.get_lock():
                url = s3_paths[file_index.value]
                file_index.value += 1
            
            bytes_results = await download_chunks(session, url, bytes_per_file, chunk_size)
            t = compile(bytes_results, activation_dtype, shape)
            write_tensor(t, buffer, writeable_tensors, readable_tensors, ongoing_downloads)


class S3RCache():
    @classmethod
    def from_credentials(self, access_key_id, secret, s3_prefix, local_cache_dir, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret)
        return S3RCache(local_cache_dir, s3_client, s3_prefix, *args, **kwargs)

    def __init__(self, 
                 s3_client, 
                 s3_prefix, 
                 bucket_name=BUCKET_NAME, 
                 device='cpu', 
                 concurrency=100, 
                 chunk_size=MB*16, 
                 buffer_size=5,
                 paths=None,
                 n_workers=1
            ) -> None:

        mp.set_start_method('spawn')

        self.s3_prefix = s3_prefix
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.device = device
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        self.paths = paths
        self._s3_paths = self._list_s3_files()

        response = self.s3_client.get_object(Bucket=bucket_name, Key=_metadata_path(s3_prefix))
        content = response['Body'].read()
        self.metadata = json.loads(content)
        self._activation_dtype = eval(self.metadata['dtype'])

        self._running_processes = []
        self.n_workers = n_workers

        self.readable_tensors = Queue(maxsize=self.buffer_size)
        self.writeable_tensors = Queue(maxsize=self.buffer_size)
        for i in range(self.buffer_size):
            self.writeable_tensors.put(i)
        self.buffer=torch.empty((self.buffer_size, *self.metadata['shape']), dtype=self._activation_dtype).share_memory_()

        self._stop = Value('b', False)
        self._file_index = Value('i', 0)
        self._ongoing_downloads = Value('i', 0)

    def sync(self):
        self._s3_paths = self._list_s3_files()

    def _reset(self):
        self._file_index.value = 0
        self._ongoing_downloads.value = 0
        self._stop.value = False

        while not self.readable_tensors.empty():
            self.readable_tensors.get()
        
        while not self.writeable_tensors.empty():
            self.writeable_tensors.get()
        for i in range(self.buffer_size):
            self.writeable_tensors.put(i)

    def _list_s3_files(self):
        if self.paths is not None:
            return self.paths

        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.s3_prefix)
        _metadata = _metadata_path(self.s3_prefix)
        paths = [f"http://{BUCKET_NAME}.s3.amazonaws.com/{obj['Key']}" for obj in response['Contents'] if obj['Key'] != _metadata]

        return sorted(paths)

    def __iter__(self):
        self._reset()

        if self._running_processes:
            raise ValueError('Cannot iterate over cache a second time while it is downloading')

        if len(self._s3_paths) > 0:
            while len(self._running_processes) < self.n_workers:
                p = Process(target=download_loop, args=(
                    self.buffer,
                    self._file_index,
                    self._s3_paths,
                    self._stop,
                    self.metadata['shape'],
                    self._activation_dtype,
                    self.readable_tensors,
                    self.writeable_tensors,
                    self._ongoing_downloads,
                    self.concurrency,
                    self.metadata['bytes_per_file'],
                    self.chunk_size
                ))
                p.start()
                self._running_processes.append(p)

        return self

    def _next_tensor(self):
        idx = self.readable_tensors.get(block=True)
        t = self.buffer[idx].clone()

        self.writeable_tensors.put(idx, block=True)

        return t
    
    def __next__(self):
        while self._file_index.value < len(self._s3_paths) or not self.readable_tensors.empty() or self._ongoing_downloads.value > 0:
            return self._next_tensor()

        if self._running_processes:
            self.stop_downloading()
        raise StopIteration

    def stop_downloading(self):
        self._file_index.value = len(self._s3_paths)
        self._ongoing_downloads.value = 0
        self._stop.value = True
        for p in self._running_processes:
            p.join()
        self._running_processes = []

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
