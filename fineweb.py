import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm 



# Set folder and dataset info
save_folder = "edu_fineweb10B"
dataset_name = "sample-10BT"
tokens_per_file = int(1e8)  # 100 million tokens in each file

data_folder = os.path.join(os.path.dirname(__file__), save_folder)
os.makedirs(data_folder, exist_ok=True)

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=dataset_name, split="train")

# Get the GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
end_token = tokenizer._special_tokens['<|endoftext|>']

# Function to convert one document to tokens
def encode_doc(doc):
    tokens = [end_token]
    tokens += tokenizer.encode_ordinary(doc["text"])
    token_array = np.array(tokens)
    assert (0 <= token_array).all() and (token_array < 2**16).all(), "Token values too large for uint16"
    return token_array.astype(np.uint16)

# Save token array to a file
def save_tokens(file_path, token_array):
    np.save(file_path, token_array)

# Use multiprocessing to encode and save dataset into multiple files
num_workers = max(1, os.cpu_count() // 2)
with mp.Pool(num_workers) as pool:
    file_index = 0
    token_buffer = np.empty((tokens_per_file,), dtype=np.uint16)
    token_pos = 0
    progress = None

    for token_list in pool.imap(encode_doc, dataset, chunksize=16):

        if token_pos + len(token_list) < tokens_per_file:
            token_buffer[token_pos:token_pos+len(token_list)] = token_list
            token_pos += len(token_list)

            if progress is None:
                progress = tqdm(total=tokens_per_file, unit="tokens", desc=f"File {file_index}")
            progress.update(len(token_list))

        else:
            file_type = "val" if file_index == 0 else "train"
            file_name = os.path.join(data_folder, f"edufineweb_{file_type}_{file_index:06d}")
            leftover = tokens_per_file - token_pos

            progress.update(leftover)
            token_buffer[token_pos:token_pos+leftover] = token_list[:leftover]
            save_tokens(file_name, token_buffer)
            file_index += 1
            progress = None

            token_buffer[0:len(token_list)-leftover] = token_list[leftover:]
            token_pos = len(token_list) - leftover

    # Save the final chunk if there are tokens left
    if token_pos != 0:
        file_type = "val" if file_index == 0 else "train"
        file_name = os.path.join(data_folder, f"edufineweb_{file_type}_{file_index:06d}")
        save_tokens(file_name, token_buffer[:token_pos])