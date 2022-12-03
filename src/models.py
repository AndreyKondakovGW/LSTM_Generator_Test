import torch
from torch import nn

class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # слой для кодирования символов в векторы [bach_size x input_size] - > [bach_size x embedding_size]
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)

        # LSTM слой [bach_size x embedding_size] - > [bach_size x hidden_size]
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)

        # полносвязный слой [bach_size x hidden_size] - > [bach_size x input_size]
        # На выходе получаем вероятности для каждого символа
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    
    def init_hidden(self, device, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))