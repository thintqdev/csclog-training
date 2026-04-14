"""CSCLog model variants."""
import collections
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import FTEncoder, IREncoder, LSTMEncoder


class _CSCLogBase(nn.Module):
    """Shared structure: FTEncoder + attention pooling + classifier head."""

    def __init__(
        self,
        input_size: int,
        com_num: int,
        ft_hid_size: int,
        lstm_hid_size: int,
        mlp_hid_size: int,
        gcn_hid_size: int,
        out_hid_size: int,
        alpha: float,
        ft_pattern: int,
        num_layers: int,
        num_keys: int,
        drop: float = 0.1,
    ):
        super().__init__()
        self.lstm_hid_size = lstm_hid_size
        self.com_num = com_num

        self.ftencoder = FTEncoder(input_size, ft_hid_size, alpha, ft_pattern)
        self.att_fc = nn.Linear(lstm_hid_size, lstm_hid_size)
        self.fc1 = nn.Linear(2 * lstm_hid_size, out_hid_size)
        self.fc2 = nn.Linear(out_hid_size, num_keys)
        self.u_att = nn.Parameter(torch.zeros(1, lstm_hid_size))
        nn.init.xavier_uniform_(self.u_att, gain=nn.init.calculate_gain("relu"))

    def _resolve(self, per_x: torch.Tensor, per_index: torch.Tensor, device):
        """Group log embeddings by component index (ordered dict)."""
        res: collections.OrderedDict = collections.OrderedDict()
        for idx in range(per_x.shape[0]):
            key = per_index[idx].item()
            if key not in res:
                res[key] = []
            res[key].append(per_x[idx].to(device))
        return res

    def _attention_net(self, x: torch.Tensor) -> torch.Tensor:
        """Dot-product attention pooling over component dimension."""
        seq_len = x.shape[1]
        re_x = x.reshape(-1, self.lstm_hid_size)
        scores = torch.mm(re_x, self.u_att.T).reshape(-1, seq_len)
        scores = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(x * scores, dim=1)
        return F.relu(self.att_fc(pooled))

    def _fuse_and_classify(self, batch_as_x: torch.Tensor, batch_ac_x: List[torch.Tensor]) -> torch.Tensor:
        ac = torch.stack(batch_ac_x).squeeze(1)  # [B, lstm_hid]
        multi_out = torch.cat((batch_as_x, ac), dim=-1)
        return self.fc2(F.relu(self.fc1(multi_out)))


class CSCLog(_CSCLogBase):
    """Full CSCLog: shared component LSTM + IREncoder GCN."""

    def __init__(self, input_size, com_num, ft_hid_size, lstm_hid_size,
                 mlp_hid_size, gcn_hid_size, out_hid_size, alpha, ft_pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__(input_size, com_num, ft_hid_size, lstm_hid_size,
                         mlp_hid_size, gcn_hid_size, out_hid_size, alpha,
                         ft_pattern, num_layers, num_keys, drop)
        self.lstm0 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)  # all-sequence
        self.lstm2 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)  # shared component
        self.irencoder = IREncoder(lstm_hid_size, mlp_hid_size, gcn_hid_size, drop, com_num)

    def forward(self, x, index, q_x, t_x):
        device = x.device
        x = self.ftencoder((x, t_x))
        batch_as_x = self.lstm0(x)

        batch_ac_x = []
        for i in range(x.shape[0]):
            res = self._resolve(x[i], index[i], device)
            ac_x = [self.lstm2(torch.stack(v).unsqueeze(0)).squeeze(0) for v in res.values()]
            ac_x = torch.stack(ac_x)
            if ac_x.shape[0] != 1:
                ac_x = self.irencoder(ac_x, list(res.keys()))
            batch_ac_x.append(self._attention_net(ac_x.unsqueeze(0)))

        return self._fuse_and_classify(batch_as_x, batch_ac_x)


class CSCLog_wo_ic(_CSCLogBase):
    """CSCLog without IREncoder (ablation: no inter-component GCN)."""

    def __init__(self, input_size, com_num, ft_hid_size, lstm_hid_size,
                 mlp_hid_size, gcn_hid_size, out_hid_size, alpha, ft_pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__(input_size, com_num, ft_hid_size, lstm_hid_size,
                         mlp_hid_size, gcn_hid_size, out_hid_size, alpha,
                         ft_pattern, num_layers, num_keys, drop)
        self.lstm0 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)
        self.lstm2 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)

    def forward(self, x, index, q_x, t_x):
        device = x.device
        x = self.ftencoder((x, t_x))
        batch_as_x = self.lstm0(x)

        batch_ac_x = []
        for i in range(x.shape[0]):
            res = self._resolve(x[i], index[i], device)
            ac_x = [self.lstm2(torch.stack(v).unsqueeze(0)).squeeze(0) for v in res.values()]
            ac_x = torch.stack(ac_x)
            batch_ac_x.append(self._attention_net(ac_x.unsqueeze(0)))

        return self._fuse_and_classify(batch_as_x, batch_ac_x)


class CSCLog_noS(_CSCLogBase):
    """CSCLog with per-component separate LSTMs (ablation: no sharing)."""

    def __init__(self, input_size, com_num, ft_hid_size, lstm_hid_size,
                 mlp_hid_size, gcn_hid_size, out_hid_size, alpha, ft_pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__(input_size, com_num, ft_hid_size, lstm_hid_size,
                         mlp_hid_size, gcn_hid_size, out_hid_size, alpha,
                         ft_pattern, num_layers, num_keys, drop)
        self.lstm0 = LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers)
        # One LSTM per component, registered as a ModuleList
        self.lstm_per_com = nn.ModuleList(
            [LSTMEncoder(ft_hid_size, lstm_hid_size, num_layers) for _ in range(com_num)]
        )
        self.irencoder = IREncoder(lstm_hid_size, mlp_hid_size, gcn_hid_size, drop, com_num)

    def forward(self, x, index, q_x, t_x):
        device = x.device
        x = self.ftencoder((x, t_x))
        batch_as_x = self.lstm0(x)

        batch_ac_x = []
        for i in range(x.shape[0]):
            res = self._resolve(x[i], index[i], device)
            ac_x = []
            for com_id, logs in res.items():
                list_x = torch.stack(logs).unsqueeze(0).to(device)
                ac_x.append(self.lstm_per_com[com_id](list_x).squeeze(0))
            ac_x = torch.stack(ac_x)
            if ac_x.shape[0] != 1:
                ac_x = self.irencoder(ac_x, list(res.keys()))
            batch_ac_x.append(self._attention_net(ac_x.unsqueeze(0)))

        return self._fuse_and_classify(batch_as_x, batch_ac_x)


class CSCLog_wo_LSTM(_CSCLogBase):
    """CSCLog with mean pooling instead of LSTM (ablation: no sequential modeling)."""

    def __init__(self, input_size, com_num, ft_hid_size, lstm_hid_size,
                 mlp_hid_size, gcn_hid_size, out_hid_size, alpha, ft_pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__(input_size, com_num, ft_hid_size, lstm_hid_size,
                         mlp_hid_size, gcn_hid_size, out_hid_size, alpha,
                         ft_pattern, num_layers, num_keys, drop)
        self.proj = nn.Linear(ft_hid_size, lstm_hid_size)
        self.irencoder = IREncoder(lstm_hid_size, mlp_hid_size, gcn_hid_size, drop, com_num)

    def forward(self, x, index, q_x, t_x):
        device = x.device
        x = self.ftencoder((x, t_x))
        batch_as_x = self.proj(torch.mean(x, dim=1))

        batch_ac_x = []
        for i in range(x.shape[0]):
            res = self._resolve(x[i], index[i], device)
            ac_x = [self.proj(torch.mean(torch.stack(v), dim=0)) for v in res.values()]
            ac_x = torch.stack(ac_x)
            if ac_x.shape[0] != 1:
                ac_x = self.irencoder(ac_x, list(res.keys()))
            batch_ac_x.append(self._attention_net(ac_x.unsqueeze(0)))

        return self._fuse_and_classify(batch_as_x, batch_ac_x)


VARIANTS = {
    "full": CSCLog,
    "wo_ic": CSCLog_wo_ic,
    "no_shared": CSCLog_noS,
    "wo_lstm": CSCLog_wo_LSTM,
}


def build_model(variant: str, **kwargs) -> _CSCLogBase:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")
    return VARIANTS[variant](**kwargs)
