import json
import boto3
import numpy as np
import time
import os
import torch
import torch.multiprocessing as mp

def upload_task(task_queue, results_queue, creds, bucket_name, shared_tensors, counts, locks, total_uploads_per_tensor):
    """
    Worker function to upload tensors to S3. Each process retrieves tensor indices
    from the task_queue, accesses the shared tensor, uploads it to S3, updates it,
    and manages the upload count.
    """
    # Each process creates its own S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=creds['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=creds['AWS_SECRET']
    )
    
    while True:
        try:
            tensor_index = task_queue.get(timeout=1)
        except Exception:
            # Timeout reached, queue might be empty
            break
        if tensor_index is None:
            break  # Sentinel value to indicate no more tasks

        lock = locks[tensor_index]

        with lock:
            count = counts[tensor_index]
            if count >= total_uploads_per_tensor:
                # No more uploads needed for this tensor
                continue

            tensor = shared_tensors[tensor_index]
            tensor_bytes = tensor.numpy().tobytes()
            object_key = f'profiling/test-tensor-{tensor_index}-{count}'  # Unique object key per upload

            # Start timing
            start_time = time.time()

            # Upload tensor bytes to S3
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=tensor_bytes,
                ContentLength=len(tensor_bytes),
                ContentType='application/octet-stream'
            )

            # End timing
            end_time = time.time()

            # Calculate time taken
            time_taken = end_time - start_time
            size_in_bytes = tensor.numel() * tensor.element_size()

            # Optionally, print progress
            print(f"Process {mp.current_process().name}: Uploaded tensor {tensor_index+1} version {count+1} of size {size_in_bytes / 1024 / 1024:.2f} MB in {time_taken:.4f} seconds.")

            # Update tensor (e.g., multiply by 2)
            tensor.mul_(2)

            # Increment upload count
            counts[tensor_index] += 1
            current_count = counts[tensor_index]

            # Put result in the results queue
            results_queue.put((time_taken, size_in_bytes))

            # If more uploads are needed for this tensor, put it back into the task queue
            if current_count < total_uploads_per_tensor:
                task_queue.put(tensor_index)

def main():
    # Use torch.multiprocessing for shared memory tensors
    mp.set_start_method('spawn', force=True)

    # Load AWS credentials from .credentials.json
    credentials_file = '.credentials.json'
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file '{credentials_file}' not found.")

    with open(credentials_file, 'r') as f:
        creds = json.load(f)

    bucket_name = 'lewington-pitsos-sache'  # Replace with your actual bucket name

    tensor_size = (257, 1000, 1000)  # Adjust size as needed

    # Number of processes to use
    num_processes = 6

    # Total number of uploads to perform
    total_uploads = 50

    # Calculate uploads per tensor
    num_tensors = 10
    total_uploads_per_tensor = total_uploads // num_tensors

    # Create shared tensors
    print("Creating tensors in shared memory...")
    tensor_create_start = time.time()
    shared_tensors = []
    for i in range(num_tensors):
        tensor = torch.rand(*tensor_size)
        tensor.share_memory_()  # Move tensor to shared memory
        shared_tensors.append(tensor)
    print(f"Created {num_tensors} tensors of size {tensor_size}.")
    tensor_create_end = time.time()
    tensor_create_time = tensor_create_end - tensor_create_start
    print(f"Time taken to create each tensor: {tensor_create_time/num_tensors:.4f} seconds")

    raise ValueError()

    # Create multiprocessing arrays for counts and locks
    counts = mp.Array('i', [0]*num_tensors)
    locks = [mp.Lock() for _ in range(num_tensors)]

    # Create multiprocessing queues
    task_queue = mp.Queue()
    results_queue = mp.Queue()

    # Enqueue initial tensor indices
    for i in range(num_tensors):
        task_queue.put(i)

    # Start overall timing
    overall_start_time = time.time()

    # Start worker processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=upload_task, args=(
            task_queue, results_queue, creds, bucket_name, shared_tensors, counts, locks, total_uploads_per_tensor))
        p.start()
        processes.append(p)

    # Collect results
    upload_times = []
    total_bytes_uploaded = 0
    completed_uploads = 0
    while completed_uploads < total_uploads:
        try:
            time_taken, size_in_bytes = results_queue.get(timeout=5)
            upload_times.append(time_taken)
            total_bytes_uploaded += size_in_bytes
            completed_uploads += 1
        except Exception:
            # Timeout reached, check if workers are still alive
            if all(not p.is_alive() for p in processes):
                break

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # End overall timing
    overall_end_time = time.time()
    overall_time_taken = overall_end_time - overall_start_time

    # Calculate mean and standard deviation
    mean_time = np.mean(upload_times)
    std_time = np.std(upload_times)

    # Calculate overall megabytes per second
    total_megabytes = total_bytes_uploaded / (1024 * 1024)
    overall_mb_per_sec = total_megabytes / overall_time_taken

    print(f"\nMean upload time per tensor: {mean_time:.4f} seconds")
    print(f"Standard deviation of upload times: {std_time:.4f} seconds")
    print(f"Total data uploaded: {total_megabytes:.2f} MB")
    print(f"Overall time taken: {overall_time_taken:.4f} seconds")
    print(f"Overall megabytes per second: {overall_mb_per_sec:.4f} MB/s")

if __name__ == '__main__':
    main()
