import torch
import torch.multiprocessing as mp
import time

def write(shared_tensor):
    start_make = time.time()
    t = torch.randn_like(shared_tensor)
    end_make = time.time()
    print(f"Time taken to make: {end_make - start_make:.2f} seconds")

    print(t[0, -2:, -2:])

    # Write directly to the shared memory tensor
    start_write = time.time()
    shared_tensor.copy_(t)
    end_write = time.time()
    print(f"Time taken to write: {end_write - start_write:.2f} seconds")

def main():
    shape = (1024, 1024, 768)

    # Create a shared tensor
    shared_tensor = torch.zeros(shape, dtype=torch.float32).share_memory_()

    p = mp.Process(target=write, args=(shared_tensor,))

    p.start()
    p.join()

    start_read = time.time()
    # Since the tensor is already in shared memory, we just access it
    tensor = shared_tensor.clone()  # Optionally create a clone to avoid affecting the original shared tensor
    end_read = time.time()
    print(f"Time taken to read: {end_read - start_read:.2f} seconds")
    print(tensor[0, -2:, -2:])

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Necessary for safely sharing CUDA tensors between processes
    main()
