import json
import time
import asyncio
import aiohttp
import randomname

async def request_chunk(session, url, start, end):
    headers = {
        "Range": f"bytes={start}-{end}"
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

# url = "https://lewington-pitsos-sache.s3.us-east-1.amazonaws.com/tensors/tensor_0.pt?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD2bv1EB6x0Kb0rz9BI2etFCZbanyeH0kNV5%2B5w2rlBAgIgCrrKDULvyafYFDjARyeUQHr6o4c6SRHkbDMY2y9FOnQq5AIIbhABGgw5MTEyMjcwOTI1MTQiDE%2B4845ukYe5fqfiMSrBAqYtuN1W5E5OJmOMuYPpVNV43sVKAzltZGCCJRa8Yyc2dtCqFtR3U%2Bm9l3sQmAWu6OYn3dPOJxgS%2Bxv9LSTTPKX1TGPPsHerrTScMwc399B06vfBchhBDT4Rikjrb5FhMr4OFGF6DDx2JgZgztIl50d2lODY%2BoYUcxa3fiXxcYv41yCJkksD%2BWCFtDrBeSyQ9zCja9WeeVeiZflxH43bOEMhPfwq%2B2KoIwrLyqUWc4KzghClXu6x2GPC3kcjRSYlnulF4JDJuj0t5OL8VITNZ9K7zN5Or5bnH517O5Bhq%2B1DPR2%2BAWi%2F0opFsfZwzvGFtiEWrl%2FCrWQyyNGwXQo1kM2aCqmUm8T2UnzkbvQYDPA24a%2B2qOMUAzG2TzgSk8wFeWL93swKhu0TazbaV1zPgcrKLeIni0qgJ269y25b%2B2t2RjDtn6y1BjqzAqWzu5RQmYuEwOZGDFJ7Dj5tR591CEO2%2BC1EJDNCp%2Fy1lfvX3qlPm%2BHsWCvc%2Fn4CYUVt0h6cOMHts1JgjyYZGdTTET7DJtw8VYoMqlc8P8lW0yEv2pxRrMwwkgT1%2FdyKxByJLRJGmH77mhQGMfwqTNpuHcI33SvajZrcKzUvi1J6CuiBck9874sJDQ2SQgeSCjhyPJxeDMWUjz3Tl3tMvhjl13G3cOS%2BrKclHynXnBK4f5DQ3nnBQ5d9V3KeGLRjHX53es4Da9vc6p1%2FZoIIr3nqCNZAztq7yn9wegAhg4hA7KFJVo9bcYf3BwNgp1BN%2BWSgE0RQ4O2rwoFfxhQl2NcPBRgLuzF2tWjQSge846o%2B8oYRYUdtFh0FWU%2Bd8zYPUdPlz9D%2Fpo0lfLJ07U75CVIZrLs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240801T043337Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIA5IKK57YRG6ZJIHON%2F20240801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a893e0bff1ada7a4b710a0b62aa6fe381c8bf4f1a23867f36c87073c42a7a8de"
url = "http://lewington-pitsos-sache.s3.us-east-1.amazonaws.com/tensors/tensor_1.pt?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD2bv1EB6x0Kb0rz9BI2etFCZbanyeH0kNV5%2B5w2rlBAgIgCrrKDULvyafYFDjARyeUQHr6o4c6SRHkbDMY2y9FOnQq5AIIbhABGgw5MTEyMjcwOTI1MTQiDE%2B4845ukYe5fqfiMSrBAqYtuN1W5E5OJmOMuYPpVNV43sVKAzltZGCCJRa8Yyc2dtCqFtR3U%2Bm9l3sQmAWu6OYn3dPOJxgS%2Bxv9LSTTPKX1TGPPsHerrTScMwc399B06vfBchhBDT4Rikjrb5FhMr4OFGF6DDx2JgZgztIl50d2lODY%2BoYUcxa3fiXxcYv41yCJkksD%2BWCFtDrBeSyQ9zCja9WeeVeiZflxH43bOEMhPfwq%2B2KoIwrLyqUWc4KzghClXu6x2GPC3kcjRSYlnulF4JDJuj0t5OL8VITNZ9K7zN5Or5bnH517O5Bhq%2B1DPR2%2BAWi%2F0opFsfZwzvGFtiEWrl%2FCrWQyyNGwXQo1kM2aCqmUm8T2UnzkbvQYDPA24a%2B2qOMUAzG2TzgSk8wFeWL93swKhu0TazbaV1zPgcrKLeIni0qgJ269y25b%2B2t2RjDtn6y1BjqzAqWzu5RQmYuEwOZGDFJ7Dj5tR591CEO2%2BC1EJDNCp%2Fy1lfvX3qlPm%2BHsWCvc%2Fn4CYUVt0h6cOMHts1JgjyYZGdTTET7DJtw8VYoMqlc8P8lW0yEv2pxRrMwwkgT1%2FdyKxByJLRJGmH77mhQGMfwqTNpuHcI33SvajZrcKzUvi1J6CuiBck9874sJDQ2SQgeSCjhyPJxeDMWUjz3Tl3tMvhjl13G3cOS%2BrKclHynXnBK4f5DQ3nnBQ5d9V3KeGLRjHX53es4Da9vc6p1%2FZoIIr3nqCNZAztq7yn9wegAhg4hA7KFJVo9bcYf3BwNgp1BN%2BWSgE0RQ4O2rwoFfxhQl2NcPBRgLuzF2tWjQSge846o%2B8oYRYUdtFh0FWU%2Bd8zYPUdPlz9D%2Fpo0lfLJ07U75CVIZrLs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240801T050854Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIA5IKK57YRG6ZJIHON%2F20240801%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e2f572a5877799e5498a69f046a9c2d68dad1dddb90db5eb863335444ec07249"
KB = 1024
MB = KB * KB


chunk_sizes = [KB * 128, KB * 512, MB * 2, MB * 8, MB * 32, MB * 128]
thread_numbers = [4, 8, 16, 32, 64]
total_sizes = [MB * 512, MB * 1024, MB * 2048, MB * 4096]


chunk_sizes = [MB * 8]
thread_numbers = [16]
total_sizes = [MB * 2048]

stats = []

async def main():
    run_name = randomname.generate('adj/', 'n/')

    print('run_name:', run_name)

    for chunk_size in chunk_sizes:
        for n_threads in thread_numbers:
            for total_size in total_sizes:
                for i in range(4):
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
