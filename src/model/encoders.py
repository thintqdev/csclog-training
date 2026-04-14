"""Neural network building blocks for CSCLog."""
import collections
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLPLayer(nn.Module):
    """Two-layer ReLU MLP."""

    def __init__(self, dmodel: int, hid_size: int, drop: float):
        super().__init__()
        self.drop = drop
        self.fc0 = nn.Linear(dmodel, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.relu(self.fc1(x))


class FTEncoder(nn.Module):
    """Fuse sentence embedding + relative timestamp → ft_hid_size vector.

    pattern 0: concatenate then linear
    pattern 1: two separate linears, then concat (default, alpha controls split)
    pattern 2: additive fusion
    """

    def __init__(self, sen_size: int, hidden_size: int, alpha: float = 0.8, pattern: int = 1):
        super().__init__()
        assert pattern in (0, 1, 2), "pattern must be 0, 1, or 2"
        self.pattern = pattern
        if pattern == 1:
            assert 0 < alpha < 1, "alpha must be in (0, 1)"
            sen_fc_size = int(hidden_size * alpha)
            time_fc_size = hidden_size - sen_fc_size
            self.sen_fc = nn.Linear(sen_size, sen_fc_size)
            self.time_fc = nn.Linear(1, time_fc_size)
        elif pattern == 0:
            self.cat_fc = nn.Linear(sen_size + 1, hidden_size)
        else:  # pattern == 2
            self.sen_fc = nn.Linear(sen_size, hidden_size)
            self.time_fc = nn.Linear(1, hidden_size)

    def forward(self, x):  # x = (sen_x [B,W,D], time_x [B,W])
        sen_x, time_x = x
        t = time_x.unsqueeze(-1)  # [B, W, 1]
        if self.pattern == 0:
            return self.cat_fc(torch.cat((sen_x, t), dim=-1))
        elif self.pattern == 1:
            return torch.cat((self.sen_fc(sen_x), self.time_fc(t)), dim=-1)
        else:
            return self.sen_fc(sen_x) + self.time_fc(t)


class LSTMEncoder(nn.Module):
    """Single LSTM returning the final hidden state."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]  # [B, hidden_size]


class IREncoder(nn.Module):
    """Implicit Relation Encoder: builds a graph over component LSTM outputs
    with GCN propagation and learned Gumbel-softmax edge weights.
    """

    def __init__(
        self,
        dmodel: int,
        mlp_hid_size: int,
        gcn_hid_size: int,
        drop: float,
        com_num: int,
    ):
        super().__init__()
        self.dmodel = dmodel
        self.drop = drop
        self.com_num = com_num

        self.edge_mlp = MLPLayer(2 * dmodel, mlp_hid_size, drop)
        self.mlp_out = nn.Linear(mlp_hid_size, 1)
        self.GCN0 = GCNConv(dmodel, gcn_hid_size)
        self.GCN1 = GCNConv(gcn_hid_size, dmodel)

    @staticmethod
    def _build_edge_index(node_indices: List[int], device: torch.device) -> torch.Tensor:
        src, dst = [], []
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                src.append(node_indices[i])
                dst.append(node_indices[j])
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    @staticmethod
    def _gumbel_softmax(x: torch.Tensor, axis: int = 1) -> torch.Tensor:
        trans = x.transpose(axis, 0).contiguous()
        soft = F.softmax(trans, dim=0)
        return soft.transpose(axis, 0)

    def forward(self, x: torch.Tensor, index: List[int]) -> torch.Tensor:
        device = x.device
        padding_x = torch.zeros(self.com_num, self.dmodel, device=device)
        padding_x[index] = x

        edge_index = self._build_edge_index(index, device)
        src, dst = edge_index[0], edge_index[1]
        edge_x = torch.cat([padding_x[src], padding_x[dst]], dim=-1)

        edge_x = self.edge_mlp(edge_x)
        edge_weight = self._gumbel_softmax(self.mlp_out(edge_x))

        out = F.relu(self.GCN0(padding_x, edge_index, edge_weight))
        out = F.dropout(out, self.drop, training=self.training)
        out = self.GCN1(out, edge_index, edge_weight)
        return out[index]
