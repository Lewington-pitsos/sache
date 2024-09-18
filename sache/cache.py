import sys
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
import asyncio
import aiohttp
import signal
from multiprocessing import Value, Process, Queue
import multiprocessing as mp
from sache.constants import BUCKET_NAME, MB, OUTER_CACHE_DIR, INNER_CACHE_DIR

STAGES = ['saved', 'shuffled']

class NoopCache():
    def __init__(self, *args, **kwargs):
        pass

    def append(self, activations):
        pass

    def finalize(self):
        pass

    def save_mean_std(self, *args, **kwargs):
        pass

class ThreadedWCache:
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

    def save_mean_std(self, mean, std):
        self.cache.save_mean_std(mean, std)

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

    def __init__(self, s3_client, run_name, save_every=1, bucket_name=BUCKET_NAME):
        self.save_every = save_every
        self.run_name = run_name
        self._in_mem = []
        self.s3_client = s3_client
        self.metadata = None
        self.bucket_name = bucket_name

    def append(self, activations):
        if self.metadata is None:
            self.metadata = _get_metadata(activations, self.save_every)

            self._save_metadata()
        else:
            if activations.shape[0] != self.metadata['batch_size']:
                print(f'Warning: batch size mismatch. Expected {self.metadata["batch_size"]}, got {activations.shape}')
            if activations.shape[-1] != self.metadata['d_in']:
                print(f'Warning: input dimension mismatch. Expected {self.metadata["d_in"]}, got {activations.shape}')
            if str(activations.dtype) != self.metadata['dtype']:
                print(f'Warning: dtype mismatch. Expected {self.metadata["dtype"]}, got {activations.dtype}')
            if len(activations.shape) == 3 and activations.shape[1] != self.metadata['sequence_length']:
                print(f'Warning: sequence length mismatch. Expected {self.metadata["sequence_length"]}, got {activations.shape}')

        self._in_mem.append(activations)

        if len(self._in_mem) == self.save_every:
            return self._save_in_mem()

        return None
    
    def _save_metadata(self):
        self.s3_client.put_object(Body=json.dumps(self.metadata), Bucket=self.bucket_name, Key=_metadata_path(self.run_name))

    def save_mean_std(self, mean, std):
        self.metadata['mean'] = mean.tolist()
        self.metadata['std'] = std.tolist()

        self._save_metadata()

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
            Bucket=self.bucket_name, 
            Key=filename, 
            Body=tensor_bytes, 
            ContentLength=len(tensor_bytes),
            ContentType='application/octet-stream'
        )

        return id

    def load(self, id):
        filename = self._filename(id)
        buffer = BytesIO()
        self.s3_client.download_fileobj(self.bucket_name, filename, buffer)
        buffer.seek(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.frombuffer(buffer.read(), dtype=torch.float32)

        return t.reshape(self.metadata['shape'])

async def download_chunks(session, url, total_size, chunk_size):
    tries_left = 5
    while tries_left > 0:
        chunks = [(i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)]
        tasks = [asyncio.create_task(request_chunk(session, url, start, end)) for start, end in chunks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        retry = False
        for response in responses:
            if isinstance(response, Exception):
                # Handle the error (e.g., log it, retry, or raise it)
                print("Error occurred:", response)
                tries_left -= 1
                retry = True
                break
            else:
                results.append(response)

        if not retry:
            return results

    return None

async def request_chunk(session, url, start, end):
    headers = {"Range": f"bytes={start}-{end}"}
    try: 
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx, etc.)
            return start, await response.read()
    except Exception as e:
        return e

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
            if bytes_results is not None:
                t = compile(bytes_results, activation_dtype, shape)
                write_tensor(t, buffer, writeable_tensors, readable_tensors, ongoing_downloads)
            else:
                print('Failed to download url', url)


class S3RCache():
    @classmethod
    def from_credentials(self, aws_access_key_id, aws_secret_access_key, *args, **kwargs):
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        return S3RCache(s3_client, *args, **kwargs)

    def __init__(self, 
                 s3_client, 
                 s3_prefix, 
                 bucket_name=BUCKET_NAME, 
                 device='cpu', 
                 concurrency=100, 
                 chunk_size=MB*16, 
                 buffer_size=2,
                 paths=None,
                 n_workers=1
            ) -> None:

        if mp.get_start_method(allow_none=True) != 'spawn':
            try:
                mp.set_start_method('spawn')
            except RuntimeError as e:
                raise RuntimeError(f'Cannot set start method to spawn. You may have created a SRCache outside of the main process. Error: {e}')

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


        signal.signal(signal.SIGTERM, self._catch_stop)
        signal.signal(signal.SIGINT, self._catch_stop)

    def _catch_stop(self, *args, **kwargs):
        print('cleaning up before process is killed')
        self._stop_downloading()
        sys.exit(0)

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
        paths = [f"http://{self.bucket_name}.s3.amazonaws.com/{obj['Key']}" for obj in response['Contents'] if obj['Key'] != _metadata]

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
                time.sleep(0.75)

        return self

    def _next_tensor(self):
        try:
            idx = self.readable_tensors.get(block=True)
            t = self.buffer[idx].clone().detach()

            self.writeable_tensors.put(idx, block=True)

            return t
        except Exception as e:
            print('exception while iterating', e)
            self._stop_downloading()
            raise StopIteration
    
    def __next__(self):
        while self._file_index.value < len(self._s3_paths) or not self.readable_tensors.empty() or self._ongoing_downloads.value > 0:
            return self._next_tensor()

        if self._running_processes:
            self._stop_downloading()
        raise StopIteration

    def finalize(self):
        self._stop_downloading()

    def _stop_downloading(self):
        print('stopping workers...')
        self._file_index.value = len(self._s3_paths)
        self._stop.value = True

        while not all([not p.is_alive() for p in self._running_processes]):
            if not self.readable_tensors.empty():
                self.readable_tensors.get()

            if not self.writeable_tensors.full():
                self.writeable_tensors.put(0)

            time.sleep(0.25)


        for p in self._running_processes:
            p.join() # still join to make sure all resources are cleaned up

        self._ongoing_downloads.value = 0
        self._running_processes = []

class ShufflingRCache():
    def __init__(self, cache, buffer_size, batch_size, d_in, dtype):
        assert buffer_size % batch_size == 0, f'buffer_size must be a multiple of batch_size, got buffer_size: {buffer_size}, batch_size: {batch_size}'

        self.cache = cache
        self.buffer = torch.empty((buffer_size, d_in), dtype=dtype)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self._current_idx = 0
        self._cache_is_empty = False

    def sync(self):
        self.cache.sync()

    def __iter__(self):
        self.cache.__iter__()
        return self

    def finalize(self):
        self.cache.finalize()

    def _full(self):
        return self._current_idx == self.buffer_size

    def _add_to_buffer(self, activations):
        flat_activations = activations.flatten(0, 1)

        if self.buffer_size % flat_activations.shape[0] != 0:
            raise ValueError(f'Buffer size {self.buffer_size} must always be divisible by flattened activations shape, but got: {flat_activations.shape}')

        next_idx = self._current_idx + flat_activations.shape[0]
        if next_idx > self.buffer_size:
            raise ValueError(f'Cannot add {flat_activations.shape[0]} activations to buffer of size {self.buffer_size}, current index is {self._current_idx}')

        self.buffer[self._current_idx:next_idx] = flat_activations

        self._current_idx = next_idx
    
    def _next(self):
        start_idx = self._current_idx - self.batch_size
        if start_idx < 0:
            raise StopIteration
        activations = self.buffer[start_idx:self._current_idx]

        self._current_idx = start_idx

        return activations

    def _flag_cache_as_empty(self):
        self._cache_is_empty = True

    def _shuffle_cache(self):
        if self._current_idx > 0:
            self.buffer[:self._current_idx] = self.buffer[torch.randperm(self._current_idx)]

    def _half_full(self):
        return self._current_idx > self.buffer_size // 2

    def __next__(self):
        if self._cache_is_empty:
            try:
                return self._next()
            except StopIteration:
                self.cache.finalize()
                raise StopIteration

        if self._half_full():
            return self._next()

        while not self._full():
            try:
                self._add_to_buffer(next(self.cache))
            except StopIteration:
                self._flag_cache_as_empty()
                break

        self._shuffle_cache()

        return self._next()

class RBatchingCache():
    def __init__(self, cache, batch_size) -> None:
        self.cache = cache
        self.batch_size = batch_size
        self.activations = None

        self._finished = False

    def finalize(self):
        self.cache.finalize()

    def sync(self):
        self.cache.sync()

    def __iter__(self):
        self._finished = False
        self.cache.__iter__()
        return self

    def mask_acts(self, acts):
        return acts

    def __next__(self):
        if self._finished:
            self.cache.finalize()
            raise StopIteration

        if self.activations is None:
            acts = next(self.cache)

            self.activations = self.mask_acts(acts)

        while self.activations.shape[0] < self.batch_size:
            try:
                self.activations = torch.cat([self.activations, self.mask_acts(next(self.cache))], dim=0)
            except StopIteration:
                self._finished = True
                if self.activations.shape[0] == 0:
                    raise StopIteration
                return self.activations

        batch = self.activations[:self.batch_size]

        self.activations = self.activations[self.batch_size:]

        return batch

def _get_metadata(activations, save_every):
    metadata = {
        'batch_size': activations.shape[0],
        'dtype': str(activations.dtype),
        'bytes_per_file': activations.element_size() * activations.numel() * save_every,
        'batches_per_file': save_every,
        'shape': (activations.shape[0] * save_every, *activations.shape[1:])
    }

    if len(activations.shape) == 3:
        metadata['sequence_length'] = activations.shape[1]
        metadata['d_in'] = activations.shape[2]
    elif len(activations.shape) == 2:
        metadata['d_in'] = activations.shape[1]
    else:
        raise ValueError(f"tried to save unexpected activations shape in metadata {activations.shape}")

    return metadata

class WCache():
    def __init__(self, run_name, save_every=1, base_dir=OUTER_CACHE_DIR):
        self.save_every = save_every

        self.outer_cache_dir = os.path.join(base_dir, run_name)
        self.cache_dir = os.path.join(self.outer_cache_dir, INNER_CACHE_DIR)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self._in_mem = []
        self.metadata = None

    def n_saved(self):
        return len(os.listdir(self.cache_dir))

    def append(self, activations):
        if self.metadata is None:
            self.metadata = _get_metadata(activations, self.save_every)

            self._save_metadata()

        self._in_mem.append(activations)

        if len(self._in_mem) >= self.save_every:
            self._save_in_mem()

    def _save_metadata(self):
        metadata_file = os.path.join(self.outer_cache_dir, 'metadata.json')

        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        

    def finalize(self):
        if self.metadata is None:
            raise ValueError("Cannot finalize cache with no metadata")
        
        if self._in_mem:
            self._save_in_mem()

    def save_mean_std(self, mean, std):
        self.metadata['mean'] = mean.tolist()
        self.metadata['std'] = std.tolist()
        
        self._save_metadata()

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
