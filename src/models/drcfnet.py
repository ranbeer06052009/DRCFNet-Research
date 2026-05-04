import torch
import torch.nn as nn
import torch.nn.functional as F

class GCFModule(nn.Module):
    def __init__(self, d_ssr):
        super(GCFModule, self).__init__()
        self.w_r = nn.Linear(d_ssr * 2, d_ssr)
        self.w_c = nn.Linear(d_ssr * 2, d_ssr)
        self.w_ct = nn.Linear(d_ssr, 1)
        self.w_ca = nn.Linear(d_ssr, 1)
        self.w_alpha = nn.Linear(d_ssr * 2, 2)
        
        self.layer_norm = nn.LayerNorm(d_ssr)
        self.ffn = nn.Sequential(
            nn.Linear(d_ssr, d_ssr),
            nn.ReLU(),
            nn.Linear(d_ssr, d_ssr)
        )
        
    def forward(self, h_main, h_comp, m):
        # h_main: e.g. ssr_t, h_comp: e.g. ssr_a, m: e.g. z_ta
        # Concat along feature dim
        concat_h = torch.cat([h_main, h_comp], dim=-1)
        
        g_r = torch.sigmoid(self.w_r(concat_h))
        g_c = torch.sigmoid(self.w_c(concat_h))
        
        c_t = torch.sigmoid(self.w_ct(h_main))
        c_a = torch.sigmoid(self.w_ca(h_comp))
        
        alpha_logits = self.w_alpha(torch.cat([h_main, m], dim=-1))
        alpha = F.softmax(alpha_logits, dim=-1)
        alpha_1 = alpha[..., 0].unsqueeze(-1)
        alpha_2 = alpha[..., 1].unsqueeze(-1)
        
        fused = alpha_1 * c_t * g_r * h_main + alpha_2 * c_a * g_c * m
        
        out = self.layer_norm(fused)
        out = self.ffn(out) + out # Residual connection on FFN
        return out


class DRCFNet(nn.Module):
    def __init__(self, dim_v=35, dim_a=74, dim_t=300, d=128, n_heads=4, dropout=0.2):
        super(DRCFNet, self).__init__()
        self.d = d
        
        # Step 1: Feature Projection
        self.proj_v = nn.Sequential(nn.Conv1d(dim_v, d, 1), nn.ReLU(), nn.Dropout(dropout))
        self.proj_a = nn.Sequential(nn.Conv1d(dim_a, d, 1), nn.ReLU(), nn.Dropout(dropout))
        self.proj_t = nn.Sequential(nn.Conv1d(dim_t, d, 1), nn.ReLU(), nn.Dropout(dropout))
        
        # Step 2: Temporal Transformer
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_v = nn.TransformerEncoder(encoder_layer_v, num_layers=1)
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_a = nn.TransformerEncoder(encoder_layer_a, num_layers=1)
        
        encoder_layer_t = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_t = nn.TransformerEncoder(encoder_layer_t, num_layers=1)
        
        # Step 3: MSR/SSR Split Projections
        self.w_ex_v = nn.Linear(d, d // 2)
        self.w_ag_v = nn.Linear(d, d // 2)
        
        self.w_ex_a = nn.Linear(d, d // 2)
        self.w_ag_a = nn.Linear(d, d // 2)
        
        self.w_ex_t = nn.Linear(d, d // 2)
        self.w_ag_t = nn.Linear(d, d // 2)
        
        # Step 4: CRE (Cross-modal Representation Encoder)
        self.cre_ta = nn.MultiheadAttention(embed_dim=d//2, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cre_tv = nn.MultiheadAttention(embed_dim=d//2, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # Step 5: GCF (Gated Controlled Fusion)
        self.gcf_ta = GCFModule(d_ssr=d//2)
        self.gcf_tv = GCFModule(d_ssr=d//2)
        
        # Step 6 & 7: Final Prediction
        # Concatenation of TA, TV, MSR_t, MSR_a, MSR_v = 5 * (d/2) = 2.5 * d
        concat_dim = 5 * (d // 2)
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1) # Regression for sentiment
        )

    def forward(self, vision, audio, text):
        # Input shapes: (Batch, Seq_Len, Dim)
        # Conv1d expects (Batch, Channels, Length), so we permute
        v = self.proj_v(vision.permute(0, 2, 1)).permute(0, 2, 1)
        a = self.proj_a(audio.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.proj_t(text.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Step 2
        h_v = self.transformer_v(v)
        h_a = self.transformer_a(a)
        h_t = self.transformer_t(t)
        
        # Step 3
        msr_v, ssr_v = self.w_ex_v(h_v), self.w_ag_v(h_v)
        msr_a, ssr_a = self.w_ex_a(h_a), self.w_ag_a(h_a)
        msr_t, ssr_t = self.w_ex_t(h_t), self.w_ag_t(h_t)
        
        # Step 4
        z_ta, _ = self.cre_ta(query=ssr_t, key=ssr_a, value=ssr_a)
        z_tv, _ = self.cre_tv(query=ssr_t, key=ssr_v, value=ssr_v)
        
        # Step 5
        out_ta = self.gcf_ta(ssr_t, ssr_a, z_ta)
        out_tv = self.gcf_tv(ssr_t, ssr_v, z_tv)
        
        # Step 6
        concatenated = torch.cat([out_ta, out_tv, msr_t, msr_a, msr_v], dim=-1)
        
        # Step 7 (Global Average Pooling over time)
        pooled = concatenated.mean(dim=1)
        output = self.fc(pooled)
        
        # Return output and necessary components for loss
        return output, {
            'msr_v': msr_v, 'ssr_v': ssr_v,
            'msr_a': msr_a, 'ssr_a': ssr_a,
            'msr_t': msr_t, 'ssr_t': ssr_t
        }
