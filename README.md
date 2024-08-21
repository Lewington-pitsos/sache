# Training a Sparse Autoencoder on GPT2 in < 40 minutes

This codebase is used to:
1. Save LLM activations to S3, i.e.
    1. Spin up a LLM
    2. Select a layer in that LLM
    3. Feed a dataset into the LLM
    4. Save the activations of that layer to S3
2. Use those LLM activations to train a sparse autoencoder quickly on a small GPU, i.e.
    1. Spin up an appropriate AWS ec2 instance
    2. Load the activations from S3
    3. Train a sparse autoencoder on those activations


## Installation

Install requirements
```bash
pip install -r requirements.txt
```

Add aws credentials to a new file called `.credentials.json` which copies the format of `example.credentials.json`

## Testing

Note: many tests will fail unless you have an internet connection so make sure you're juiced up.

```bash
python -m pytest
```

## Generating activations

```bash
python scripts/generate.py --bucket_name my_epic_bucket
```

Will start the generation process, keep track of the "run name" as this is the prefix under which the activations will be saved in s3. Note that they are saved in a janky format (raw bytes) to make them quicker to load on the other end. Once you save them you can't simply load them using `torch.load`, you need to use `torch.frombuffer` (see `sache/cache.py`). The activations are stored in ~3GB files consisting of (batch_size, sequence_length, hidden_dim) = (1024, 1024, 768) tensors.


Example activations for `gpt2-small` on 678,428,672 tokens are available [here](). <<<<<<<<<<<<<<>>>>>>>>>>>>>>

## Deploying a server to AWS

Make sure you have terraform installed and then edit `server.tf` so that the `aws_key_pair.public_key` points to a local public key (which will allow you to access the server via SSH.)

Then from the root of this project

```bash
terraform -chdir=./terraform/environments/production apply
```

This is important since loading activations from s3 will be super slow unless your sever is deployed inside the same region as your s3 bucket king. Other instance types will mostly also be incredibly slow, see ec2 instance [throughput limits]() <<<<<<<<<<<<>>>>>>>>>>>>. Once the instance is deployed you can SSH into it and start training the SAE.

Note: the `aws_instance.ami` has been carefully chosen to make nvidia actually work, but it will only work inside `us-east-1`. If you want to deploy elsewhere you will need to find the equivalent ami for that region.

## Training a SAE

You should only do this inside AWS in the same region as your bucket or else it will be horrifically slow. 

```bash
python scripts/train_sae.py --run_name merciless_citadel --use_wandb --log_bucket bucket_full_of_karpathy_fanart
```

Using the settings specified in the terraform and loading the `merciless_citadel` activations you will achieve something like 420 mbps throughput, which equates to 300,000,000 tokens in ~35 minutes.

Note that by default we log metrics and histograms locally, to wandb and also s3. See `sache/analysis.ipynb` for how to read the logged data.
