import torch
import torch.multiprocessing as mp
import time
from multiprocessing import shared_memory
import numpy as np


def write(shm_name, shape):
    start_make = time.time()
    t = torch.rand(shape)
    end_make = time.time()
    print(f"Time taken to make: {end_make - start_make:.2f} seconds")

    print(t[0, -2:, -2:])


    existing_shm = shared_memory.SharedMemory(name=shm_name)
    data = t.numpy().tobytes()
    # Write directly to the shared memory tensor
    start_write = time.time()
    existing_shm.buf[:] = data
    print(type(data))
    
    # Close the shared memory block
    existing_shm.close()
    end_write = time.time()
    print(f"Time taken to write: {end_write - start_write:.2f} seconds")

def main():
    shape = (1024, 1024, 768)
    size = shape[0] * shape[1] * shape[2] * 4

    # Create a shared tensor
    # shared_tensor = torch.zeros(shape, dtype=torch.float32).share_memory_()

    shm = shared_memory.SharedMemory(create=True, size=size)

    p = mp.Process(target=write, args=(shm.name, shape))

    p.start()
    p.join()

    start_read = time.time()

    data = shm.buf[:]
    print(type(data))
    t = torch.frombuffer(data, dtype=torch.float32).view(shape)
    end_read = time.time()
    print(f"Time taken to read: {end_read - start_read:.2f} seconds")
    print(t[0, -2:, -2:])

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Necessary for safely sharing CUDA tensors between processes
    main()
