import boto3
from boto3.s3.transfer import TransferConfig
import io
import time
import logging
import sys
import os
import json
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import BUCKET_NAME

logging.basicConfig(filename='cruft/download_speed.log', level=logging.INFO, format='%(asctime)s %(message)s')

with open('.credentials.json') as f:
    credentials = json.load(f)

KB = 1024
MB = KB * KB

def download_file(s3_client, key, bucket_name):
    start_time = time.time()

    # Download the file and store it in a buffer
    buffer = io.BytesIO()
    cfg=TransferConfig(
        # use_threads=False,
        multipart_threshold = 16 * MB, # the threshold of file size above which we do a multipart download
        max_concurrency = 30, # number of threads????
        multipart_chunksize = 8 * MB, # the amount of bytes to request from s3 in each "part" of a multipart download
        max_io_queue = 100000000,
        io_chunksize = 8 * MB,
    )
    cfg.max_in_memory_download_chunks = 100
    s3_client.download_fileobj(
        bucket_name, 
        key, 
        buffer, 
        Config=cfg
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate the file size in MB and download speed in MB/s
    file_size_mb = len(buffer.getvalue()) / (1024 * 1024)
    download_speed_mbps = file_size_mb / elapsed_time

    # Log the download speed
    logging.info(f'Downloaded {key} from {bucket_name}: {download_speed_mbps:.2f} MB/s')

    return file_size_mb, elapsed_time, buffer


class Reader():
    def __init__(self):
        self.responses = []
        self.reading_thread = threading.Thread(target=self._read)
        self.stop_reading = False
        self.reading_thread.start()

    def _read(self):
        while True:
            if self.stop_reading:
                break
            if len(self.responses) == 0:
                time.sleep(0.05)
            else:
                for buffer in self.responses:
                    buffer.seek(0)
                    buffer.read()
                self.responses = []

    def stop(self):
        self.stop_reading = True
        self.reading_thread.join()
        

def main(n_files, num_threads):
    # List of S3 file paths (bucket_name/key)
    s3_files = [f'tensors/tensor_{i}.pt' for i in range(n_files)]

    total_size_mb = 0
    total_time_sec = 0
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'], use_ssl=False)
    
    r = Reader()
    for key in s3_files:
        file_size_mb, elapsed_time, buffer = download_file(s3_client, key, BUCKET_NAME)

        r.responses.append(buffer)

        total_size_mb += file_size_mb
        total_time_sec += elapsed_time

    # Calculate and display total download speed
    if total_time_sec > 0:
        total_download_speed_mbps = total_size_mb / total_time_sec
        print(f'Total time: {total_time_sec:.2f} seconds')
        print(f'Total download speed: {total_download_speed_mbps:.2f} MB/s')
    else:
        print('Total time is zero, cannot calculate download speed.')
    r.stop()

if __name__ == '__main__':
    n_files = 2
    num_threads = 1
    main(n_files, num_threads)
