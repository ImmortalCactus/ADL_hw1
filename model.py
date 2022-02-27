from typing import Dict

import torch
from torch.nn import Embedding

import torch.nn as nn
class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.bidirect_multiplier = 2 if bidirectional else 1

        # GRU
        hidden0 = torch.zeros(self.bidirect_multiplier*self.num_layers, 1, self.hidden_size)
        self.init_hidden = nn.Parameter(hidden0)
        self.gru = nn.GRU(300, self.hidden_size, num_layers=self.num_layers, bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.bidirect_multiplier*hidden_size, num_class)
        self.act_fn = nn.LeakyReLU(0.1)
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        out = self.embed(batch)
        out, h = self.gru(out, self.init_hidden.repeat(1,batch.size(dim=0),1))
        out = out[:,-1,:].view(out.size(dim=0),-1)
        out = self.dropout_layer(out)
        out = self.linear(out)
        out = self.act_fn(out)
        return out
