# coding: utf-8

import argparse
import logging
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

from load_data import get_data_loaders
from model import LSTMTextClassifier, DANTextClassifier, CNNTextClassifier, DenseTextClassifier, AttnTextClassifier
from utils import logging_config, get_device
from torchtext.vocab import pretrained_aliases
import torch.nn as nn
import torch.nn.functional as F

import sys

parser = argparse.ArgumentParser(description='Train a (short) text classifier - via convolutional or other standard architecture')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data', default='data/ar_gk_train.tsv')
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default='data/ar_gk_test.tsv')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')

#argument for choosing the model
parser.add_argument('--dan', action='store_true', help='Use a DAN encoder', default=False)
parser.add_argument('--dense', action='store_true', help='Use a Dense encoder', default=False)
parser.add_argument('--cnn', action='store_true', help='Use a CNN encoder', default=False)
parser.add_argument('--lstm', action='store_true', help='Use an LSTM layer', default=False)
parser.add_argument('--attn', action='store_true', help='Use an attention layer', default=False)

#hyperparameters for training
parser.add_argument('--epochs', type=int, default=100, help='Upper epoch limit')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=8)
parser.add_argument('--seq_length', type=int, help='Max sequence length', default=16)
parser.add_argument('--clip', type=float, default=10.0, help='Gradient clipping value')

#hyperparameters common to all models
parser.add_argument('--embedding_size', type=int, default=50, help='Embedding size (if random)')
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0)

#hyperparameters specific to some models
parser.add_argument('--dense_dan_layers', type=str, default='100', help='List if integer dense unit layers (for DAN or Dense)')
parser.add_argument('--filter_sizes', type=str, default='3,4', help='List of integer filter sizes (for CNN)')
parser.add_argument('--num_filters', type=int, default=100, help='Number of filters (of each size) (for CNN)')
parser.add_argument('--num_conv_layers', type=int, default=1, help='Number of convolutional/pool layers (for CNN)')
parser.add_argument('--hidden_size', type=int, default=100, help='Dimension size for hidden states (for LSTM)')
parser.add_argument('--num_lstm_layers', type=int, default=1, help='Number of LSTM layers (for LSTM)')
parser.add_argument('--num_heads', type=int, default=50, help='Number of attention heads (for Attn)')
parser.add_argument('--num_attn_layers', type=int, default=1, help='Number of attention layers (for LSTM)')


args = parser.parse_args()
loss_fn = torch.nn.CrossEntropyLoss()

def get_model(num_classes, vocab_size=0, embedding_size=0, pretrained_vectors=None):

    # Embedding looksups take one-hot integers as inputs
    # and return outputs of dimensionality `embedding_size`.
    emb_input_dim, emb_output_dim = vocab_size, embedding_size

    # Instantiate the right model
    if args.dan:
        print("Using DAN")
        dense_units = [int(x) for x in args.dense_dan_layers.split(",")]
        model = DANTextClassifier(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            num_classes=num_classes,
            dr=args.dropout,
            dense_units=dense_units,
        )
    elif args.dense:
        print("Using Dense")
        dense_units = [int(x) for x in args.dense_dan_layers.split(",")]
        model = DenseTextClassifier(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            num_classes=num_classes,
            dr=args.dropout,
            dense_units=dense_units,
        )
    elif args.cnn:
        print("Using CNN")
        filters = [ int(x) for x in args.filter_sizes.split(',') ]
        model = CNNTextClassifier(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            filter_widths=filters,
            num_classes=num_classes,
            dr=args.dropout,
            num_conv_layers=args.num_conv_layers,
            num_filters=args.num_filters
        )
    elif args.lstm:
        print("Using LSTM")
        model = LSTMTextClassifier(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            hidden_size=args.hidden_size,
            num_lstm_layers=args.num_lstm_layers,
            num_classes=num_classes,
            dr=args.dropout,
        )
    else:
        print("Using Attn")
        dense_units = [int(x) for x in args.dense_dan_layers.split(",")]
        model = AttnTextClassifier(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            num_heads=args.num_heads,
            num_attn_layers=args.num_attn_layers,
            num_classes=num_classes,
            dr=args.dropout
        )

    return model


def train_classifier(model, train_loader, val_loader, test_loader, num_classes, device):
    
    trainer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (label, ids, mask) in enumerate(train_loader):
            output = model(ids, mask)
            loss = loss_fn(output, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            trainer.step() # update weights based on gradient
            epoch_loss += loss.item()
        # Output the epoch loss into our log file
        logging.info("Epoch {} loss = {}".format(epoch + 1, epoch_loss))
    
        tr_acc = evaluate(model, train_loader, device) # compute accuracy

        # Now we'll log the metrics
        logging.info("TRAINING Acc = {}".format(tr_acc))
        val_acc = evaluate(model, val_loader, device)
        logging.info("VALIDATION Acc = {}".format(val_acc))
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(model, 'model.pt')
            # evaluate the current best model on test data
            if test_loader is not None:
                tst_acc = evaluate(model, test_loader, device)
                logging.info("TEST Acc = {}".format(tst_acc))
                print("\n")


def evaluate(model, dataloader, device):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for i, (label, ids, mask) in enumerate(dataloader):
            output = model(ids, mask)
            pred = torch.argmax(output, dim=1)
            total_correct += (pred == label).float().sum()
            total += pred.shape[0]
    acc = (total_correct / total)
    return acc

if __name__ == "__main__":

    args.dan = False
    args.dense = False
    args.cnn = False
    args.lstm = False
    args.attn = True
    
    # Set up logging
    logging_config(args.log_dir, "train", level=logging.INFO)

    # We assign an integer ID to each objective
    label_map = {"O": 0, "NAME": 1, "STATE": 2, "UNIT": 3, "QUANTITY": 4,
                 "SIZE": 5, "TEMP": 6, "DF": 7}

    # Get the data loaders
    vocab, train_loader, val_loader, test_loader = get_data_loaders(
        args.train_file,
        args.test_file,
        args.batch_size,
        args.seq_length,
        label_map
    )

    pretrained, embedding_size = None, args.embedding_size
        
    # Now we'll instantiate the model itself.
    model = get_model(

        len(label_map),
        vocab_size=len(vocab)+3,
        embedding_size=args.embedding_size,
        pretrained_vectors=pretrained,
    )
    # vocab_size=len(vocab)+3 to make room for out-of-vocab and padding id, when index starts at 1

    # Get the device we're using and move the model to it
    device = get_device()
    model.to(device)

    # Actually train the model given the data
    train_classifier(
        model, train_loader, val_loader, test_loader, len(label_map), device=device
    )
