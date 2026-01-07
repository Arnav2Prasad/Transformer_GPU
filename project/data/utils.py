
import requests
import tiktoken
import numpy as np


'''
# _______________ DATASET _________________
def tokenize_and_save():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        text = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        return # Exit the function if download fails

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)

    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]

    data_splits = {'train': train_data, 'val': val_data}
    for split, data in data_splits.items():
        file_path = f'{split}.bin'
        with open(file_path, 'wb') as f:
            f.write(data.tobytes())


'''

# _______________ DATASET _________________
def tokenize_and_save():
    # TinyStories dataset URL from Hugging Face
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        text = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the TinyStories dataset: {e}")
        
        # Try alternative URL
        print("Trying alternative URL...")
        url = "https://raw.githubusercontent.com/roneneldan/TinyStories/master/TinyStoriesV2-GPT4-train.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text
        except requests.exceptions.RequestException as e2:
            print(f"Error downloading from alternative URL: {e2}")
            return  # Exit the function if all downloads fail

    # If you also want to include the validation split, you can download it too:
    # val_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    # response_val = requests.get(val_url)
    # text_val = response_val.text
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize the training data
    tokens = enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    
    # Split into training and validation sets (90/10 split)
    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]
    
    # Alternative: If you downloaded separate validation data, use that instead
    # tokens_val = enc.encode(text_val)
    # val_data = np.array(tokens_val, dtype=np.uint16)
    
    data_splits = {'train': train_data, 'val': val_data}
    for split, data in data_splits.items():
        file_path = f'{split}.bin'
        with open(file_path, 'wb') as f:
            f.write(data.tobytes())
        print(f"Saved {split} split with {len(data)} tokens to {file_path}")
    
    # Optional: Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Training tokens: {len(train_data)}")
    print(f"Validation tokens: {len(val_data)}")
    print(f"Vocabulary size: {enc.n_vocab}")