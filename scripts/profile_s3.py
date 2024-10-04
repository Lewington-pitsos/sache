import json
import boto3
import numpy as np
import time
import os
import multiprocessing

def upload_task(args):
    index, creds, bucket_name, tensor_size = args

    # Each process creates its own S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=creds['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=creds['AWS_SECRET']
    )

    # Each process creates its own tensor
    tensor = np.random.rand(*tensor_size)
    tensor_bytes = tensor.tobytes()
    object_key = f'profiling/test-tensor-{index}'  # Unique object key for each upload

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
    size_in_bytes = tensor.nbytes

    # Optionally, print progress
    print(f"Task {index+1}: Uploaded tensor of size {size_in_bytes / 1024 / 1024:.2f} MB in {time_taken:.4f} seconds.")

    return time_taken, size_in_bytes

def main():
    # Load AWS credentials from .credentials.json
    credentials_file = '.credentials.json'
    if not os.path.exists(credentials_file):
        raise FileNotFoundError(f"Credentials file '{credentials_file}' not found.")

    with open(credentials_file, 'r') as f:
        creds = json.load(f)

    if 'AWS_ACCESS_KEY_ID' not in creds or 'AWS_SECRET' not in creds:
        raise KeyError("Credentials file must contain 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET' keys.")

    # Define S3 bucket name
    bucket_name = 'lewington-pitsos-sache'  # Replace with your actual bucket name

    # Create S3 client in the main process to check/create the bucket
    s3_client = boto3.client(
        's3',
        aws_access_key_id=creds['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=creds['AWS_SECRET']
    )

    # Ensure the bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except s3_client.exceptions.NoSuchBucket:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Created bucket: {bucket_name}")

    # Define tensor size
    tensor_size = (257, 1000, 1000)  # Adjust size as needed

    # Number of processes (adjust as needed)
    num_processes = 6
    total_uploads = 50

    # Create a multiprocessing Queue to collect results
    result_queue = multiprocessing.Queue()

    # Start overall timing
    overall_start_time = time.time()

    task_args = [
        (i, creds, bucket_name, tensor_size)
        for i in range(total_uploads)
    ]

    # Collect results
    upload_times = []
    total_bytes_uploaded = 0

    # Use multiprocessing Pool to distribute tasks among processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the upload_task function to the task arguments
        results = pool.map(upload_task, task_args)

    # Process results
    for time_taken, size_in_bytes in results:
        upload_times.append(time_taken)
        total_bytes_uploaded += size_in_bytes
    # End overall timing
    overall_end_time = time.time()
    overall_time_taken = overall_end_time - overall_start_time

    # Calculate mean and standard deviation
    mean_time = np.mean(upload_times)
    std_time = np.std(upload_times)

    # Calculate overall megabytes per second
    total_megabytes = total_bytes_uploaded / (1024 * 1024)
    overall_mb_per_sec = total_megabytes / overall_time_taken

    print(f"\nMean upload time per process: {mean_time:.4f} seconds")
    print(f"Standard deviation of upload times: {std_time:.4f} seconds")
    print(f"Total data uploaded: {total_megabytes:.2f} MB")
    print(f"Overall time taken: {overall_time_taken:.4f} seconds")
    print(f"Overall megabytes per second: {overall_mb_per_sec:.4f} MB/s")

if __name__ == '__main__':
    main()
