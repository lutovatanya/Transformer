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
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(d_model, ntoken)
        self.d_model = d_model


    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output



if __name__ == "__main__":

    #encoder_par = count_parameters(nn.TransformerEncoderLayer(d_model=1, nhead=1, dropout=0.5))

    a = TransformerModel(ntoken=1, d_model=1, nhead=1, d_hid=128, nlayers=1, dropout=0.5)
    src = torch.randn(1200, 10, 1)
    out = a(src)
    print(out)
    print(out.size())

    """

    encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dropout=0.5)
    src1 = torch.randn(1200, 10, 1)
    out1 = encoder_layer(src1)
    print(out1.size())

    m = nn.Linear(1, 4)
    input = out1
    output = m(input)
    print(output.size())

    encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, d_hid=128, dropout=0.5)
    src2 = torch.randn(1400, 10, 1)
    out2 = encoder_layer(src2)
    print(out2.size())

    m2 = nn.Linear(1, 4)
    input2 = out2
    output2 = m2(input2)
    print(output2.size())

    """


