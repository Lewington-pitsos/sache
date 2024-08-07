import torch
import torch.multiprocessing as mp
import time

def write(shared_tensor, idx):
    start_make = time.time()
    t = torch.randn_like(shared_tensor[idx])
    end_make = time.time()
    print(f"Time taken to make: {end_make - start_make:.2f} seconds")

    print(t[0, -2:, -2:])

    # Write directly to the shared memory tensor
    start_write = time.time()
    shared_tensor[idx, :] = t
    end_write = time.time()
    print(f"Time taken to write: {end_write - start_write:.2f} seconds")

def main():
    start_share = time.time()
    shape = (1024, 1024, 768)

    # Create a shared tensor
    shared_tensor = torch.empty((3, *shape), dtype=torch.float32).share_memory_()
    end_share = time.time()
    print(f"Time taken to share: {end_share - start_share:.2f} seconds")

    pool = []
    for i in range(3):
        p = mp.Process(target=write, args=(shared_tensor,0))
        p.start()
        pool.append(p)

    for p in pool:
        p.join()


    start_read = time.time()
    # Since the tensor is already in shared memory, we just access it
    tensor = shared_tensor[0].clone()  # Optionally create a clone to avoid affecting the original shared tensor
    end_read = time.time()
    print(f"Time taken to read: {end_read - start_read:.2f} seconds")
    print(tensor[0, -2:, -2:])
    print(f"Total time: {end_read - start_share:.2f} seconds")

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Necessary for safely sharing CUDA tensors between processes
    main()
