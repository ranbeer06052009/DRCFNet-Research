# fusion/mulg.py
import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention:
    source attends to target
    """

    def __init__(self, d_src, d_tgt):
        super().__init__()
        self.W = nn.Linear(d_tgt, d_src, bias=False)

    def forward(self, X_src, X_tgt):
        """
        X_src : (B, T_src, d_src)
        X_tgt : (B, T_tgt, d_tgt)
        """
        X_tgt_proj = self.W(X_tgt)                         # (B, T_tgt, d_src)
        scores = torch.bmm(X_src, X_tgt_proj.transpose(1, 2))
        alpha = torch.softmax(scores, dim=-1)
        return torch.bmm(alpha, X_tgt)                     # (B, T_src, d_tgt)

    # def forward(self, X_src, X_tgt):
    # # Shape guards
    #   if X_src.dim() == 1:
    #     X_src = X_src.unsqueeze(0).unsqueeze(1)
    #   elif X_src.dim() == 2:
    #     X_src = X_src.unsqueeze(0)

    #   if X_tgt.dim() == 1:
    #     X_tgt = X_tgt.unsqueeze(0).unsqueeze(1)
    #   elif X_tgt.dim() == 2:
    #     X_tgt = X_tgt.unsqueeze(0)

    # # Empty memory case
    #   if X_tgt.size(1) == 0:
    #     return X_src

    #   X_tgt_proj = self.W(X_tgt)                 # (B, T_tgt, d_src)
    #   scores = torch.bmm(X_src, X_tgt_proj.transpose(1, 2))
    #   alpha = torch.softmax(scores, dim=-1)
    #   return torch.bmm(alpha, X_tgt)

class CrossModalGRU(nn.Module):
    """
    Cross-modal GRU block
    """

    def __init__(self, d_src, d_tgt, hidden_dim):
        super().__init__()
        self.attn = CrossModalAttention(d_src, d_tgt)
        self.gru = nn.GRU(d_src + d_tgt, hidden_dim, batch_first=True)

    def forward(self, X_src, X_tgt):
        """
        X_src : (B, T_src, d_src)
        X_tgt : (B, T_tgt, d_tgt)
        """
        X_tgt_att = self.attn(X_src, X_tgt)
        X = torch.cat([X_src, X_tgt_att], dim=-1)
        out, _ = self.gru(X)
        return out


class MulGFusion(nn.Module):
    """
    MulG fusion module (ASYNC)
    """

    def __init__(self, d_l=50, d_a=50, d_v=50, hidden_dim=50):
        super().__init__()

        # language as source
        self.l_a = CrossModalGRU(d_l, d_a, hidden_dim)
        self.l_v = CrossModalGRU(d_l, d_v, hidden_dim)

        # audio as source
        self.a_l = CrossModalGRU(d_a, d_l, hidden_dim)
        self.a_v = CrossModalGRU(d_a, d_v, hidden_dim)

        # vision as source
        self.v_l = CrossModalGRU(d_v, d_l, hidden_dim)
        self.v_a = CrossModalGRU(d_v, d_a, hidden_dim)

    def forward(self, modalities):
        """
        modalities = [X_v, X_a, X_l]
        """
        X_v, X_a, X_l = modalities

        hl = self.l_a(X_l, X_a) + self.l_v(X_l, X_v)
        ha = self.a_l(X_a, X_l) + self.a_v(X_a, X_v)
        hv = self.v_l(X_v, X_l) + self.v_a(X_v, X_a)

        # take last timestep
        hl = hl[:, -1, :]
        ha = ha[:, -1, :]
        hv = hv[:, -1, :]

        return torch.cat([hl, ha, hv], dim=1)   # (B, 3*hidden_dim)

