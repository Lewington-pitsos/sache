import time
import io
import torch
import numpy as np
import os


# def generate_random_tensor(size_gb):
#     """Generates a random tensor of the specified size in GB."""
#     tensor_size = round((size_gb * 1024**3)) // 4  # float32 has 4 bytes
#     tensor = torch.randn(tensor_size // (1024 * 1024), 1024 * 1024)
#     return tensor

# tensor = generate_random_tensor(5)

# print(tensor.shape)
# print(tensor[:4, :4])
# tensor_bytes = tensor.numpy().tobytes()

# # save to file

# with open('cruft/bytes-tensor.pt', 'wb') as f:
#     f.write(tensor_bytes)


# torch.save(tensor, 'cruft/normal-tensor.pt')






def load_chunks_file(filename):
    buffers = []
    file_size = os.path.getsize(filename)  # Get the total file size
    bytes_read = 0

    with open(filename, 'rb') as f:
        while bytes_read < file_size:
            # Calculate remaining size to read to ensure we don't exceed the total file size
            remaining_size = min(1280 * 4, file_size - bytes_read)
            chunk = f.read(remaining_size)

            if not chunk:
                break
            buffers.append(chunk)
            bytes_read += len(chunk)
            
    return buffers


def load_normal_tensor():
    buffers = load_chunks_file('cruft/normal-tensor.pt')
    start = time.time()

    combined_bytes = b''.join(buffers)

    buffer = io.BytesIO(combined_bytes)

    t = torch.load(buffer, map_location='cuda', weights_only=True)

    end = time.time()
    print('normal time:', end - start)
    print(t.shape)
    print(t[:4, :4])


def load_bytes_tensor():
    buffers = load_chunks_file('cruft/bytes-tensor.pt')
    start = time.time()

    combined_bytes = b''.join(buffers)

    n = np.frombuffer(combined_bytes, dtype=np.float32)
    reshaped = n.reshape(1280, 1024 * 1024)

    t = torch.from_numpy(reshaped)

    end = time.time()
    print('bytes time:', end - start)
    print(t.shape)
    print(t[:4, :4])


def load_bytes_tensor2():
    buffers = load_chunks_file('cruft/bytes-tensor.pt')
    start = time.time()

    combined_bytes = b''.join(buffers)

    t = torch.frombuffer(combined_bytes, dtype=torch.float32)
    t = t.reshape(1280, 1024 * 1024)

    end = time.time()
    print('bytes time:', end - start)
    print(t.shape)
    print(t[:4, :4])

load_bytes_tensor2()
load_normal_tensor()
load_bytes_tensor()