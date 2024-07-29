import torch
import time

# Define the size of the tensor
tensor_size = 5 * 1024 * 1024 * 1024  # 5 GB
tensor_shape = (tensor_size // 4,)  # Using float32, so 4 bytes per element

# Create a 5 GB tensor
tensor = torch.randn(tensor_shape, dtype=torch.float32)

# File path to save the tensor
file_path = 'tensor.pt'

# Profile writing the tensor to disk
start_write = time.time()
torch.save(tensor, file_path)
end_write = time.time()

# Calculate write throughput
write_time = end_write - start_write
write_throughput = tensor_size / write_time / (1024 ** 3)  # GB/s

print(f"Write Time: {write_time:.2f} seconds")
print(f"Write Throughput: {write_throughput:.2f} GB/s")

# Profile loading the tensor from disk
start_read = time.time()
loaded_tensor = torch.load(file_path, weights_only=True)
end_read = time.time()

# Calculate read throughput
read_time = end_read - start_read
read_throughput = tensor_size / read_time / (1024 ** 3)  # GB/s

print(f"Read Time: {read_time:.2f} seconds")
print(f"Read Throughput: {read_throughput:.2f} GB/s")


# Scores for local macbook

# Write Time: 5.84 seconds
# Write Throughput: 0.86 GB/s
# Read Time: 3.93 seconds
# Read Throughput: 1.27 GB/s