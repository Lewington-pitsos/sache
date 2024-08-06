import boto3
import json
import time
import asyncio
import sys 
import os
import multiprocessing as mp


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.cache import S3RCache
from sache.constants import *


paths = [
    [
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/00a06c26-6835-4009-9556-012ac978b8b9.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0105dfce-4be1-4a11-93c2-09605c1d6514.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/01755f8a-28e6-480f-8a33-f90dd89aec58.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/02152822-79cc-48d8-89a2-f21976fef415.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/034bbdbf-25eb-4536-b8c4-43b470323a08.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0363baf8-a57d-4205-8eb2-e334a9928386.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/038c8f08-d94e-474b-bbca-5b02ce8df90b.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/03eaef78-f53d-42fe-893b-1deac8616640.saved.pt",
],
[
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/03fc601b-bfed-4618-98a9-8140cd37476d.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/04262ec9-25c3-452f-b861-4c041f4888e4.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/042ff908-7e74-4491-93ca-038b72420712.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/05896c95-ef5f-4c62-982e-29b1b5ccbc44.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/05c82a66-c4b2-4be2-97be-fbb06a49fb5f.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/05e4e503-11d4-4cd7-b84b-032d6a392eaf.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0612917b-c644-41b4-84e3-b88375ba8ce9.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/068864f7-fd5b-4f3a-9382-d0916577769b.saved.pt",
],
[
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/06e2e013-6842-4466-905c-6c8ef06190bc.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/075a33f6-8740-4f09-993f-059f2a6d0e05.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0787ae59-db1a-4344-9723-ffb2a21584a0.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/085e229e-7f2d-42c5-9b54-f8bd61fe1e01.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/08d71f2c-62ad-4c23-bf42-3128de526a15.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/08ebeba0-dcba-4c3b-bee0-5193d50ddbc9.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/09523640-ebcb-4815-88cf-e2f94215db2b.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/09722dec-9249-42d4-8fa7-efd8b2d90d0d.saved.pt",
],
[
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0a07e117-ab2a-4654-983b-2b873bdbc71b.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0a4c15f9-dff3-4fb0-98fc-8e8950b6c3a7.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0a9e3136-3e15-4ab0-8042-5f86bd482a9a.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0ac0bf22-1c38-425b-b96c-779b9df57906.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0ae1410f-ff90-47eb-8ff7-708c080bfc5f.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0aff04cd-bb15-4cf1-8624-e02c586575cb.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0c7fdbe8-366a-4d38-802d-0bf1dc084e93.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0d2e5c40-096f-4bd3-b090-3d314cd1ff12.saved.pt",
],
[
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0dcb99bc-d216-4dfb-9225-be118890fd85.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0e4c4a2e-ba01-4401-b0a0-06645c42befc.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0e8ef967-f34a-4397-b0c8-efb52ed83e6e.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0e932b12-49f7-4959-a3c9-bf03f720e52e.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0f711363-0079-4a6b-9a68-45ff57475f5c.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0fb92330-cfec-4ff9-85ad-d73b5f2773cd.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0feeef8c-45fd-4f9a-b3a8-bcc3f1c82357.saved.pt",
"http://lewington-pitsos-sache.s3.amazonaws.com/merciless-citadel/0ffb8deb-5b8a-4789-8c2a-02611a0d7e37.saved.pt",
],
]

def main():
    mp.set_start_method('fork')
    pool = []

    for i in range(2):
        pool.append(mp.Process(target=download, args=(paths[i],)))
    
    for p in pool:
        p.start()

    s = time.time()
    print('started')

    for p in pool:
        p.join()

    print('finished')
    e = time.time()
    elapsed = e - s
    print(f"Total time taken: {elapsed:.2f} seconds")
    print(f"MB per second: {2 * 8 * 3000 / elapsed:.2f}")

def download(paths):
    asyncio.run(_download(paths))

async def _download(paths):
    with open('.credentials.json') as f:
        credentials = json.load(f)
    s3_client = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET'])

    cache = S3RCache(s3_client, 'merciless-citadel', 'lewington-pitsos-sache', chunk_size=MB * 8, concurrency=500, paths=paths)
    
    iter(cache)
    
    total_size = cache.metadata['bytes_per_file']
    for i in range(8):
        start = time.time()

        t = next(cache)

        end = time.time()


        print(f"Time taken: {end - start:.2f} seconds")
        print(f"MB Downloaded: {round(total_size / MB)}, MB per second: {round(total_size / MB) / (end - start):.2f}")

    cache.stop_downloading()


if __name__ == "__main__":
    main()
