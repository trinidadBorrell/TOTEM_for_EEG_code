import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .transformer import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class ScaleEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0):
        super(ScaleEncoding, self).__init__()

        self.proj = nn.Linear(2, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            x: (S, B, d_model)
            scale: (S, B, 2)
        """
        return self.dropout(x + self.proj(scale))


class TimeEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        seq_len: int = 5000,
        dropout: float = 0.0,
        batch_first: bool = False,
        norm_first: bool = False,
        return_weights: bool = False, 
    ):
        super(TimeEncoder, self).__init__()
        self.model_type = "Time"
        self.d_model = d_model
        self.return_weights = return_weights

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len + 1)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # self._reset_parameters()
        self.apply(self._init_weights)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape (seq_len, batch_size, d_in)
        Returns:
            y: tensor of shape (batch_size, d_model)
        """
        if self.has_linear_in:
            x = self.linear_in(x)
        # pos encoder
        x = self.pos_encoder(x)
        x, weights = self.transformer_encoder(x)  # (seq_len, batch, d_model)

        x = self.pos_encoder(x)
        x = x.mean(dim=0)  # (batch, d_model)

        if self.return_weights: 
            return x, weights
        else: 
            return x


class SensorEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        seq_len: int = 5000,
        dropout: float = 0.0,
        batch_first: bool = False,
        norm_first: bool = False,
        return_weights: bool = False, 
        scale: bool = True
    ):
        super(SensorEncoder, self).__init__()
        self.model_type = "Sensor"
        self.d_model = d_model
        self.return_weights = return_weights
        self.scale = scale

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)
        if self.scale:
            self.scale_encoder = ScaleEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # TODO implementing Georgia change
        self.proj = nn.Linear(d_model, d_model)

        # self._reset_parameters()
        self.apply(self._init_weights)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: tensor of shape (seq_len, batch_size, d_in)
            scale: tensor of shape (seq_len, batch_size, 2)
        Returns:
            y: tensor of shape (batch_size, seq_out_len)
        """

        if self.has_linear_in:
            x = self.linear_in(x)
        # scale encoding
        if self.scale: 
            x = self.scale_encoder(x, scale)
        # pos encoder
        x = self.pos_encoder(x)
        x, weights = self.transformer_encoder(x)  # (seq_len, batch, d_model)

        x = torch.permute(x, (1, 0, 2))  # (batch, seq_len, d_model)
        # TODO adding change Georgia suggestion
        x = x.mean(dim=1)
        # x = x.flatten(start_dim=1)  # (batch, seq_len * d_model)
        x = self.proj(x)

        if self.return_weights: 
            return x, weights
        else: 
            return x


class SensorTimeEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int, 
        nheadt: int,
        nheads: int,
        d_hid: int,
        nlayerst: int,
        nlayerss: int,
        seq_lent: int = 5000,
        seq_lens: int = 5000,
        dropout: float = 0.0,
        d_out: int = 1,
        return_weights: bool = False, 
        scale: bool = True
    ):
        super(SensorTimeEncoder, self).__init__()
        self.model_type = "SensorTime"
        self.return_weights = return_weights

        self.timeenc = TimeEncoder(
            d_in=d_in,
            d_model=d_model,
            nhead=nheadt,
            d_hid=d_hid,
            nlayers=nlayerst,
            seq_len=seq_lent,
            dropout=dropout,
            return_weights=return_weights, 
        )

        self.senorenc = SensorEncoder(
            d_in=d_model,
            d_model=d_model,
            nhead=nheads,
            d_hid=d_hid,
            nlayers=nlayerss,
            seq_len=seq_lens,
            dropout=dropout,
            return_weights=return_weights, 
            scale=scale
        )

        self.classifier = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, T, S, d_in)
            scale: tensor of shape (B, S, 2)
        Returns:
            y: tensor of shape (B, 1)
        """

        B, T, S, dim = x.shape
        x = torch.permute(x, (1, 0, 2, 3))  # (T, B, S, d_in)

        # prepare input for time encoder
        x = x.flatten(start_dim=1, end_dim=2)  # (T, B * S, d_in)
        if self.return_weights:
            y, time_weights = self.timeenc(x)  # (B * S, d_model)
        else: 
            y = self.timeenc(x)  # (B * S, d_model)

        # prepare input to sensor encoder
        y = y.reshape(B, S, y.shape[-1])  # (B, S, d_model)
        y = torch.permute(y, (1, 0, 2))  # (S, B, d_model)
        scale = torch.permute(scale, (1, 0, 2))  # (S, B, 2)
        if self.return_weights: 
            z, sensor_weights = self.senorenc(y, scale)  # (B, d_model)
        else: 
            z = self.senorenc(y, scale)  # (B, d_model)

        z = self.classifier(z)

        if self.return_weights: 
            return z, time_weights, sensor_weights
        else: 
            return z


def Lout(Lin, kernel, stride=1, padding=0, dilation=1):
    """
    Returns the length of the tensor after a conv layer
    From: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    return math.floor((Lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, dropout=0.0):
        super(SimpleMLP, self).__init__()

        self.nlayers = len(hidden_dims)

        layers = []
        dim = in_dim
        for i in range(self.nlayers):
            layer = nn.Linear(dim, hidden_dims[i])
            layers.append(layer)
            dim = hidden_dims[i]
        self.fcs = nn.ModuleList(layers)
        self.fc_out = nn.Linear(dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc_out(x)

        return x


# Source: https://github.com/torcheeg/torcheeg/blob/v1.1.0/torcheeg/models/cnn/eegnet.py#L15-L126
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    """
    Args:
        chunk_size (int): T
        num_electrodes (int): S
        F1 (int): The filter number of block 1,
        F2 (int): The filter number of block 2
        D (int): The depth multiplier (number of spatial filters)
        num_classes (int): The number of classes to predict
        kernel_1 (int): The filter size of block 1
        kernel_2 (int): The filter size of block 2
        dropout (float): probability of dropout
    """

    def __init__(
        self,
        chunk_size: int = 151,
        num_electrodes: int = 60,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_1: int = 64,
        kernel_2: int = 16,
        dropout: float = 0.25,
        num_classes: int = 1, 
    ):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernel_1),
                stride=1,
                padding=(0, self.kernel_1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.num_electrodes, 1),
                max_norm=1,
                stride=1,
                padding=(0, 0),
                groups=self.F1,
                bias=False,
            ),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernel_2),
                stride=1,
                padding=(0, self.kernel_2 // 2),
                bias=False,
                groups=self.F1 * self.D,
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                1,
                padding=(0, 0),
                groups=1,
                bias=False,
                stride=1,
            ),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout),
        )

        self.lin = nn.Linear(self.feature_dim(), self.num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): (B, 1, S, T)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)

        return x
