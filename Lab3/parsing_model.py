import torch
import torch.nn as nn
import torch.nn.functional as F


class ParsingModel(nn.Module):

    def __init__(self, words_num, label_num, dropout: float = 0.5, embedding_size: int = 128, hidden_size: int = 512):
        """ 
        Initialize the parser model. You can add arguments/settings as you want, depending on how you design your model.
        NOTE: You can load some pretrained embeddings here (If you are using any).
              Of course, if you are not planning to use pretrained embeddings, you don't need to do this.
        """
        super(ParsingModel, self).__init__()
        self.word_embedding = nn.Embedding(words_num, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(embedding_size * 48, hidden_size)
        self.out = nn.Linear(hidden_size, label_num)

    def forward(self, t):
        """
        Input: input tensor of tokens -> SHAPE (batch_size, n_features)
        Return: tensor of predictions (output after applying the layers of the network
                                 without applying softmax) -> SHAPE (batch_size, n_classes)
        """
        x = self.word_embedding(t)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        logits = self.out(x)
        return logits

    def reg(self):
        l2_hidden = (self.hidden_layer.weight**2).sum()**0.5
        l2_out = (self.out.weight**2).sum()**0.5
        return l2_hidden + l2_out