for i in {00000..00127}; do wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en/resolve/main/.part-$i-00478b7a-941e-4176-b569-25f4be656991-c000.snappy.parquet.crc; done


for i in {00004..00010}; do wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en/resolve/main/part-$i-00478b7a-941e-4176-b569-25f4be656991-c000.snappy.parquet; done



for i in {00011..00022}; do wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/datasets/laion/laion2B-multi-joined-translated-to-en/resolve/main/part-$i-00478b7a-941e-4176-b569-25f4be656991-c000.snappy.parquet; done


aws s3 sync ./cruft/ViT-3mil-topkk-32-experts-None_1aaa89 s3://vit-sae-switch/images/ViT-3mil-topkk-32-experts-None_1aaa89
aws s3 sync ./cruft/ViT-3mil-topkk-32-experts-8_5d073c s3://vit-sae-switch/images/ViT-3mil-topkk-32-experts-8_5d073c
aws s3 sync ./cruft/ViT-3mil-relu-l1-0.0001_ed4f74 s3://vit-sae-switch/images/ViT-3mil-relu-l1-0.0001_ed4f74