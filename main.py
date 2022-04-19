import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Params: {total_params}")
    return total_params


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, num_classes: int, n_channels: int, dropout: float = 0.5 ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(n_channels, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src: Tensor) -> Tensor:
        src = torch.permute(src, (2, 0, 1))
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = torch.permute(output, (1, 2, 0))
        return output

if __name__ == "__main__":

    model = TransformerModel(d_model=64, nhead=8, d_hid=132, nlayers=8,
                             num_classes=4, n_channels=1, dropout=0.5)
    par = count_parameters(model)

    src = torch.randn(10, 1, 1200)
    print(src.size())
    out = model(src)
    print(out.size())

    src2 = torch.randn(10, 1, 1400)
    print(src2.size())
    out2 = model(src2)
    print(out2.size())

