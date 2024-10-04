import json
import boto3
import numpy as np
import time
import os

credentials_file = '.credentials.json'
with open(credentials_file, 'r') as f:
    creds = json.load(f)

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=creds['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=creds['AWS_SECRET']
)

# Define S3 bucket name
bucket_name = 'lewington-pitsos-sache'  # Replace with your actual bucket name

# Ensure the bucket exists
try:
    s3_client.head_bucket(Bucket=bucket_name)
except s3_client.exceptions.NoSuchBucket:
    s3_client.create_bucket(Bucket=bucket_name)
    print(f"Created bucket: {bucket_name}")

# Create a sample tensor (numpy array)
tensor_size = (257, 1000, 1000)  # Adjust size as needed
tensor = np.random.rand(*tensor_size)

# Convert tensor to bytes
tensor_bytes = tensor.tobytes()

# List to store upload times
upload_times = []

# Perform 10 uploads
for i in range(10):
    object_key = f'profiling/test-tensor-{i}'  # Unique object key for each upload

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
    upload_times.append(time_taken)

    print(f"Upload {i+1}: Uploaded tensor of size {tensor.nbytes / 1024 / 1024} megabytes in {time_taken:.4f} seconds.")

# Calculate mean and standard deviation
mean_time = np.mean(upload_times)
std_time = np.std(upload_times)

print(f"\nMean upload time: {mean_time:.4f} seconds")
print(f"Mean megabytes per second: {tensor.nbytes / 1024 / 1024 / mean_time:.4f} MB/s")
print(f"Standard deviation of upload times: {std_time:.4f} seconds")
