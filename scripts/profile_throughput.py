import os
import torch
import time
import concurrent.futures

# Define the size of the tensor
tensor_size = 5 * 1024 * 1024 * 1024  # 5 GB
tensor_shape = (tensor_size // 4,)  # Using float32, so 4 bytes per element

# Create a 5 GB tensor
tensor = torch.randn(tensor_shape, dtype=torch.float32)

if not os.path.exists('cruft'):
    os.makedirs('cruft', exist_ok=True)

# File path to save the tensor
file_path = 'cruft/tensor.pt'

# Profile writing the tensor to disk
start_write = time.time()
torch.save(tensor, file_path)
end_write = time.time()

# Calculate write throughput
write_time = end_write - start_write
write_throughput = tensor_size / write_time / (1024 ** 3)  # GB/s

print(f"Write Time: {write_time:.2f} seconds")
print(f"Write Throughput: {write_throughput:.2f} GB/s")

# Function to load tensor from disk
def load_tensor():
    return torch.load(file_path, weights_only=True)

# Number of concurrent reads
num_reads = 4

# Profile loading the tensor from disk with concurrent reads
start_read = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=num_reads) as executor:
    futures = [executor.submit(load_tensor) for _ in range(num_reads)]
    loaded_tensors = [f.result() for f in concurrent.futures.as_completed(futures)]
end_read = time.time()

# Calculate read throughput
read_time = end_read - start_read
read_throughput = tensor_size * num_reads / read_time / (1024 ** 3)  # GB/s

print(f"Read Time: {read_time:.2f} seconds")
print(f"Read Throughput (concurrent reads): {read_throughput:.2f} GB/s")


# Write Time: 5.84 seconds
# Write Throughput: 0.86 GB/s
# Read Time: 3.93 seconds
# Read Throughput: 1.27 GB/s

# Write Time: 15.20 seconds
# Write Throughput: 0.33 GB/s
# Read Time: 5.03 seconds
# Read Throughput (concurrent reads): 3.98 GB/s