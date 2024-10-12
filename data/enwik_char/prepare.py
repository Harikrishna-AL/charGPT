import os
import pickle
import requests
import numpy as np
import sys
import zipfile

# download the tiny shakespeare dataset
url = "http://mattmahoney.net/dc/enwik8.zip"
__file__ = os.path.abspath(__file__)

def download_file(url):
    file_name = url.split('/')[-1]
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(file_name):
        data = requests.get(url)
        with open(file_name, 'wb') as f:
            f.write(data.content)
    else:
        print(f"file {file_name} already exists")

def clean_wikipedia_text(text):
    import re
    match = re.search(r'<text.*?>(.*)</text>', text, re.DOTALL)
    data = match.group(1)
    # Remove all XML tags using a regular expression
    text = re.sub(r'<[^>]+>', '', data)
    # Remove #REDIRECT lines
    text = re.sub(r'#REDIRECT \[\[.*?\]\]', '', text)
    
    # Remove metadata (timestamps, user info, IDs)
    text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '', text)  # Timestamps
    text = re.sub(r'(\bcur_id=\d+|\b\d{1,9})', '', text)  # IDs and numbers
    text = re.sub(r'[A-Za-z]+\s\d{4,6}', '', text)  # User info

    # Remove links in double brackets but keep their text
    text = re.sub(r'\[\[([^\|\]]+\|)?([^\]]+)\]\]', r'\2', text)  # Keep the link text

    # Remove templates in curly brackets but keep text inside
    text = re.sub(r'\{\{([^\|\}]+\|)?([^\}]+)\}\}', r'\2', text)

    # Remove reference tags and content inside <ref></ref>
    text = re.sub(r'<ref.*?>.*?</ref>', '', text)

    # Remove HTML-like special characters (e.g., &amp;, &quot;, etc.)
    text = re.sub(r'&[a-z]+;', '', text)

    # Remove URLs in brackets
    text = re.sub(r'\[http[^\]]*\]', '', text)

    # Remove equal signs '=' and apostrophes '''
    text = re.sub(r'[=\'"]+', '', text)

    text = text.strip()

    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines into one
    text = re.sub(r'[ \t]+', ' ', text)

    return text

#main thread
def main():
    print("downloading the dataset...")
    download_file(url)
    print("downloaded the dataset")

    print("Extracting and preparing the dataset...")
    zip_path = os.path.join(os.path.dirname(__file__), 'enwik8.zip')

    data = zipfile.ZipFile(zip_path).read('enwik8')
    data = [chr(s) for s in data]
    data = ''.join(data)
    data = clean_wikipedia_text(data)
    print(data[:200])

    print(f"length of dataset in characters: {len(data):,}")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("done!")

main()