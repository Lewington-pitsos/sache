import json
import time
import asyncio
import aiohttp
import randomname
import random

async def request_chunk(session, url, start, end):
    rand_offset = random.randint(0, 100)
    start = max(0, start - rand_offset)
    end = end - rand_offset
    headers = {
        "Range": f"bytes={start}-{end}",
    }
    async with session.get(url, headers=headers) as response:
        return await response.read()

async def download_chunks(url, total_size, chunk_size, n_threads):
    chunks = [(i, min(i + chunk_size - 1, total_size - 1)) for i in range(0, total_size, chunk_size)]
    
    connector = aiohttp.TCPConnector(limit=n_threads)  # Adjust the limit as needed
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(request_chunk(session, url, start, end)) for start, end in chunks]
        results = await asyncio.gather(*tasks)
    
    return results


url = 'http://lewington-pitsos-sache.s3.amazonaws.com/tensors/tensor_2.pt'
KB = 1024
MB = KB * KB

# chunk_sizes = [KB * 512, MB * 2, MB * 8, MB * 32, MB * 128]
# thread_numbers = [4, 8, 16, 32, 64]
# total_sizes = [MB * 512, MB * 1024, MB * 2048, MB * 4096]

chunk_sizes = [MB * 16]
thread_numbers = [36]
total_sizes = [MB * 4096]

stats = []

async def main():
    run_name = randomname.generate('adj/', 'n/')

    print('run_name:', run_name)

    for chunk_size in chunk_sizes:
        for n_threads in thread_numbers:
            for total_size in total_sizes:
                for i in range(40):
                    start = time.time()

                    responses = await download_chunks(url, total_size, chunk_size, n_threads)

                    end = time.time()

                    stats.append({
                        "param_hash": str(round(chunk_size / MB, 2)) + '_' + str(n_threads) + '_' + str(round(total_size / MB, 2)),
                        "time": end - start,
                        "download_speed": total_size / (end - start),
                        "MB per second": round(total_size / MB) / (end - start),
                        "total_size": total_size,
                        "chunk_size": chunk_size,
                        "n_threads": n_threads
                    })

                    with open(f"cruft/{run_name}-stats.json", "w") as f:
                        json.dump(stats, f)

                    print(f"Time taken: {end - start:.2f} seconds")
                    print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

if __name__ == "__main__":
    asyncio.run(main())
