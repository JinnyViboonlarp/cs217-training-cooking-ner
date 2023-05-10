# codeing: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import sys


class DANTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        dense_units=[100, 100]
    ):
        super(DANTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        modulelist = []
        for i in dense_units:
            modulelist.append(nn.Sequential(
            nn.Dropout(dr),
            nn.LazyLinear(i), # input size is inferred
            nn.ReLU()
        ))
        self.feedforwards = nn.ModuleList(modulelist)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.LazyLinear(num_classes)
        

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]

        #seperate the embedding of the word (to do NER on) from that of the whole sentence
        embedded_split = torch.split(embedded, split_size_or_sections=1, dim=1)
        embedded_word = embedded_split[0] # [batch_size, 1, embedded_size]
        embedded_sentence = torch.cat(embedded_split[1:], dim=1) # [batch_size, seq_len-1, embedded_size]

        #concat the embedding of the word (to do NER on) with the averaged sentence embedding
        avg_embedded = embedded_sentence.mean(dim=1) # [batch_size, embedded_size]
        embedded_word = torch.squeeze(embedded_word) # [batch_size, embedded_size]
        x = torch.cat((embedded_word, avg_embedded), dim=1) # [batch_size, 2*embedded_size]

        for feedforward in self.feedforwards:
            x = feedforward(x)
            
        x = self.dropout(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x


class DenseTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_classes=2,
        dr=0.2,
        dense_units=[100, 100]
    ):
        super(DenseTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        modulelist = []
        for i in dense_units:
            modulelist.append(nn.Sequential(
            nn.Dropout(dr),
            nn.LazyLinear(i), # input size is inferred
            nn.ReLU()
        ))
        self.feedforwards = nn.ModuleList(modulelist)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.LazyLinear(num_classes)
        

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x = torch.flatten(embedded, start_dim=1, end_dim=2) # [batch_size, seq_len*embedded_size] 

        for feedforward in self.feedforwards:
            x = feedforward(x)

        x = self.dropout(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x

class CNNTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0,
                 num_classes=2,
                 dr=0.2,
                 filter_widths=[3,4],
                 num_filters=100,
                 num_conv_layers=3,
                 intermediate_pool_size=3, **kwargs):
        super(CNNTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        # create the list "convblocks" consisting of only first convblock
        convblocks = [nn.ModuleList([
            nn.Dropout(dr),
            nn.ModuleList([nn.Conv1d(in_channels=emb_output_dim, out_channels=num_filters,
                                     kernel_size=filter_width) for filter_width in filter_widths]),
            nn.ReLU(),
            nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
        ])]
        # append subsequent convblocks
        for i in range(num_conv_layers - 1):
            convblocks.append(nn.ModuleList([
                nn.Dropout(dr),
                nn.ModuleList([nn.Conv1d(in_channels=(num_filters*len(filter_widths)), out_channels=num_filters,
                                         kernel_size=filter_width) for filter_width in filter_widths]),
                nn.ReLU(),
                nn.MaxPool1d(intermediate_pool_size, ceil_mode=True)
            ]))
        self.convblocks = nn.ModuleList(convblocks)
        self.globalpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.Linear(in_features=(num_filters*len(filter_widths)), out_features=num_classes)
         

    def forward(self, data, mask):
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x = embedded.transpose(1, 2).contiguous() # [batch_size, embedded_size, seq_len]
        for i in range(len(self.convblocks)):
            convblock = self.convblocks[i] #first block's input size: [batch_size, embedded_size, seq_len]
            x = convblock[0](x) #dropout
            xs = [conv(x) for conv in convblock[1]] # conv: [batch_size, num_filters, (seq_len - filter_width + 1)]
            xs = [convblock[2](x) for x in xs] #relu
            xs = [convblock[3](x) for x in xs] #maxpool: [batch_size, num_filters, (seq_len - filter_width + 1)/pool_size]
            """
            xs = [conv(x) for conv in convblock[0]] # [batch_size, num_filters, (seq_len - filter_width + 1)]
            xs = [convblock[1](x) for x in xs] #relu
            xs = [convblock[2](x) for x in xs] #maxpool: [batch_size, num_filters, (seq_len - filter_width + 1)/pool_size]
            """
            # ensure that all tensors have the same size in the 2nd dim
            min_output_length = min([x.shape[2] for x in xs])
            xs = [torch.narrow(input=x, dim=2, start=0, length=min_output_length) for x in xs]
            # concat along the num_filter dimension
            x = torch.cat(xs, dim=1) # [batch_size, len(filter_width) * num_filters, (seq_len - filter_width + 1)/pool_size]
            # with each subsequent iteration, the 2nd dim would be divided down by pool_size
            
        x = self.globalpool(x) # [batch_size, len(filter_width) * num_filters, 1]
        x = torch.squeeze(x)   # [batch_size, len(filter_width) * num_filters]
        x = self.dropout(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x
   

class LSTMTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    hidden_size : int
        Dimension size for hidden states within the LSTM
    num_lstm_layers: int
        Number of LSTM layers
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    """

    def __init__(
        self, emb_input_dim=0, emb_output_dim=0, hidden_size=100, num_lstm_layers=1, num_classes=2, dr=0.2
    ):
        super(LSTMTextClassifier, self).__init__()

        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.lstm = nn.LSTM(input_size=emb_output_dim, hidden_size=hidden_size,
                            num_layers=num_lstm_layers, batch_first=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.LazyLinear(num_classes)
        

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x, _ = self.lstm(embedded) # [batch_size, seq_len, hidden_size]
        x = x.transpose(1,2) # [batch_size, hidden_size, seq_len]
        x = self.maxpool(x) # [batch_size, hidden_size, 1]
        x = torch.squeeze(x) # [batch_size, hidden_size]
        x = self.dropout(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x


class AttnTextClassifier(nn.Module):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_heads : int
        Number of attention heads
    num_attn_layers: int
        Number of attention layers
    num_classes : int, default 2
        Number of categories in classifier output
    dr : float, default 0.2
        Dropout rate
    """

    def __init__(
        self,
        emb_input_dim=0,
        emb_output_dim=0,
        num_heads=10,
        num_attn_layers=1,
        num_classes=2,
        dr=0.2
    ):
        super(AttnTextClassifier, self).__init__()
        self.emb_input_dim = emb_input_dim
        self.emb_output_dim = emb_output_dim
        
        self.embedding = nn.Embedding(emb_input_dim, emb_output_dim)
        self.attentions = nn.ModuleList([nn.MultiheadAttention(emb_output_dim, num_heads=num_heads)] * num_attn_layers)
        self.dropout = nn.Dropout(dr)
        self.projection = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data, mask):
        
        data = (data * mask)
        embedded = self.embedding(data) # [batch_size, seq_len, embedded_size]
        x = embedded

        for attention in self.attentions:
            x, _ = attention(query=x, key=x, value=x) # [batch_size, seq_len, embedded_size]

        x = torch.flatten(x, start_dim=1, end_dim=2) # [batch_size, seq_len*embedded_size]
        x = self.dropout(x)
        x = self.projection(x) # [batch_size, num_classes]
        return x
