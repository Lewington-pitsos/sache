import time
from s5cmdpy import S5CmdRunner
runner = S5CmdRunner()


start = time.time()
txt_s3_uris = [
    "s3://lewington-pitsos-sache/tensors/tensor_0.pt",
    "s3://lewington-pitsos-sache/tensors/tensor_1.pt",
    "s3://lewington-pitsos-sache/tensors/tensor_2.pt",
]
runner.download_from_s3_list(txt_s3_uris, 'cruft')

end = time.time()

print(f"Time taken: {round(end - start, 2)} seconds")