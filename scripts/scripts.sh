for i in {00000..00127}; do wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en/resolve/main/.part-$i-00478b7a-941e-4176-b569-25f4be656991-c000.snappy.parquet.crc; done


for i in {00004..00010}; do wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en/resolve/main/part-$i-00478b7a-941e-4176-b569-25f4be656991-c000.snappy.parquet; done
