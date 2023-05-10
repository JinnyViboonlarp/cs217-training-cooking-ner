# coding: utf-8

import io
import orjson
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from utils import get_device

import sys

#========== Do the necessary loading and setting things up ================

vocab_path = 'vocab.txt'

def load_vocab(vocab_path):
    # this function is used to load the pre-saved vocab
    vocab = {}
    with open(vocab_path, 'r') as fh_in:
        lines = fh_in.readlines()
    for line in enumerate(lines):
        tokens = line[1].strip().split()
        if(len(tokens) >= 2):
            (i, word) = (tokens[0], " ".join(tokens[1:]))
            vocab[word] = int(i)
    return vocab

vocab = load_vocab(vocab_path)
out_of_vocab_index = (len(vocab)+1)
padding_index = (len(vocab)+2)

def text_pipeline(words):
    return [vocab.get(word, out_of_vocab_index) for word in words]

#==========================================================================

def create_vocab():
    # this function is used only once to create vocab file to be re-loaded later
    train_file = 'data/ar_gk_train.tsv'
    data = get_data(train_file)
    
    vocab_list = sum([words for (words, tags) in data], [])
    vocab = sorted(set(vocab_list))

    with open('vocab.txt', 'w') as fh_out:
        fh_out.writelines([(str(i)+'\t'+str(word)+'\n') for i, word in enumerate(vocab, start=1)])
    

def get_data(filename):
    # this function is used to read data in the correct format
    data = []
    ingredients = []
    tags = []
    with open(filename) as file:
        for line in file:
            token = line.strip().split('\t')
            if(len(token)<2): #line break
                if(len(ingredients)>0):
                    data.append((ingredients, tags))
                ingredients = []
                tags = []
            else:
                ingredients.append(token[0])
                tags.append(token[1])
    return data


def get_loader(data, label_pipeline, batch_size, max_len, shuffle=False):
    """Takes a dataset and pipelines returning a PyTorch dataloader ready for use in training."""

    # Use helper function to infer what device to use (use GPU if available)
    device = get_device()

    def collate_batch(batch):

        label_list, text_list, mask_list = [], [], []
        # Iterate over all (label, text) tuples
        for (_text, _label) in batch:
            # Process label and text using pipelines
            label = label_pipeline(_label)
            orig_text = text_pipeline(_text)
            for i,w in enumerate(orig_text):
                # Copy the token for the NER to the start of the text.
                text = [w] + orig_text
                # If too short, pad to max_len.
                text = text + ([padding_index] * max_len)
                # If sentence too long, truncate to max_len.
                text = text[:max_len]
                # Create a mask tensor indicating where padding is present.
                pad_mask = [0 if i==padding_index else 1 for i in text]
        
                label_list.append(label[i])
                text_list.append(text)
                mask_list.append(pad_mask)

        label_list = torch.tensor(label_list)
        text_list = torch.tensor(text_list)
        mask_list = torch.tensor(mask_list)

        return label_list.to(device), text_list.to(device), mask_list.to(device)

    # At the end, we create a DataLoader object and return it
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )


def get_data_loaders(train_file, test_file, batch_size, max_len, label_map):

    # Take the file paths {train, test}_file and load the data contained in the files.
    train_data = get_data(train_file)
    test_data = get_data(test_file)

    # Split test_data into val_data and a proper test_data
    cutoff = int(len(test_data)/2)
    val_data, test_data = test_data[:cutoff], test_data[cutoff:]

    """
    train_data has 6111 sentences, 37214 tokens
    val_data has 1093 sentences, 5705 tokens
    test_data has 1094 sentences, 6965 tokens
    """
    
    # e.g. train_data[0] = (['4', 'cloves', 'garlic'], ['QUANTITY', 'UNIT', 'NAME'])

    # Callable for data loader that gets a label from `label_map` for each data point x
    label_pipeline = lambda x: [label_map.get(w, 0) for w in x]
    # text_pipeline is already defined as a function
    # e.g. text_pipeline(['4', 'cloves', 'garlic']) = [18, 63, 28]

    # Get {train,val,test} data loaders
    train_dataloader = get_loader(
        train_data, label_pipeline, batch_size, max_len, shuffle=True
    )
    val_dataloader = get_loader(
        val_data, label_pipeline, batch_size, max_len, shuffle=False
    )
    test_dataloader = get_loader(
        test_data, label_pipeline, batch_size, max_len, shuffle=False
    )
    return vocab, train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    # called only once to create vocab file to be re-loaded later
    create_vocab()
