python scripts/vit_generate.py --run_name "test3" --n_samples=20000 --batch_size=1024 --batches_per_cache=2 --n_hooks=4 --full_sequence
python scripts/vit_generate.py --run_name "test4" --n_samples=20000 --batch_size=512 --batches_per_cache=2 --n_hooks=8 --full_sequence
python scripts/vit_generate.py --run_name "test5" --n_samples=20000 --batch_size=512 --batches_per_cache=2 --n_hooks=None --full_sequence


python scripts/vit_generate.py --run_name "multilayer" --n_samples=20000 --batch_size=1024 --batches_per_cache=11 --full_sequence --input_tensor_shape="(257,1024)" 