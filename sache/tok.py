
# sneakily stolen from https://github.com/EleutherAI/sae/blob/9e43ce3e39dcba003df96af8c9449bc5b5937b83/sae/data.py (seemed
# excessive to include the whole thing just for this)

import math
from multiprocessing import cpu_count
from typing import Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

def chunk_and_tokenize(
    data: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    batch_size: int = 2048,
    override_eos_token: str = None,
) -> Dataset:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.

    Returns:
        The chunked and tokenized dataset.
    """

    if tokenizer.eos_token is None and override_eos_token is None:
        raise ValueError("Tokenizer must have an EOS token to use chunk_and_tokenize, or you must pass in an override")

    def _tokenize_fn(x: dict[str, list], leftovers: list=[]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if (not return_final_batch) and len(output["input_ids"][-1]) != chunk_size:
            # we do not pad so if the last batch is smaller than the required
            # batch size we either lengthen it using leftover batches or put
            # it in the basket of leftovers
            final_chunk = output["input_ids"].pop()
            
            while len(final_chunk) < chunk_size:
                if len(leftovers) == 0:
                    leftovers.append(final_chunk)
                    break
                
                leftover = leftovers.pop()
                final_chunk.extend([tokenizer.eos_token_id] + leftover)
            else:
                new_leftover = final_chunk[chunk_size:]
                final_chunk = final_chunk[:chunk_size]
                output["input_ids"].append(final_chunk)
                
                if len(new_leftover) > 0:
                    leftovers.append(new_leftover)

            output = {k: v[:len(output['input_ids'])] for k, v in output.items()}


        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        fn_kwargs={} if return_final_batch else {"leftovers": []}
    )
    return data.with_format(format, columns=["input_ids"])


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names