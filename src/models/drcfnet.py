import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSTCNProjection(nn.Module):
    """Multi-Scale Temporal Convolutional Projection"""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, 5, padding=2)
        self.skip = nn.Conv1d(in_channels, out_channels // 4, 1)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, T)
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        sk = self.skip(x)
        
        out = torch.cat([c1, c3, c5, sk], dim=1) # (B, out, T)
        out = out.permute(0, 2, 1) # (B, T, out)
        out = self.norm(out)
        return self.dropout(F.relu(out))

class MultiHeadGraphFusion(nn.Module):
    """Multi-Head Graph Attention Fusion"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, nodes):
        # nodes is a list of 5 tensors, each (B, T, d)
        x = torch.stack(nodes, dim=2) # (B, T, N, d)
        B, T, N, d = x.size()
        
        # Project and split into heads
        q = self.w_q(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3) # (B, T, h, N, dk)
        k = self.w_k(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3)
        v = self.w_v(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, T, h, N, N)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v) # (B, T, h, N, dk)
        out = out.transpose(2, 3).contiguous().view(B, T, N, d)
        out = self.w_o(out)
        
        # Residual and Norm
        out = self.layer_norm(x + out)
        return [out[:, :, i, :] for i in range(N)]

class GatedSplit(nn.Module):
    """Gated MSR/SSR Disentanglement Split"""
    def __init__(self, d_model):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_model // 2)
        self.w_msr = nn.Linear(d_model, d_model // 2)
        self.w_ssr = nn.Linear(d_model, d_model // 2)
        
    def forward(self, h):
        gate = torch.sigmoid(self.w_gate(h))
        msr = self.w_msr(h) * gate
        ssr = self.w_ssr(h) * (1 - gate)
        return msr, ssr

class GCFModule(nn.Module):
    """Advanced Gated Controlled Fusion with Feature-wise Attention"""
    def __init__(self, d_model):
        super(GCFModule, self).__init__()
        self.w_c = nn.Linear(d_model, d_model)
        self.w_a = nn.Linear(d_model, d_model) # Feature-wise weights
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, h):
        # Confidence gate
        c = torch.sigmoid(self.w_c(h))
        # Feature-wise attention (softmax over feature dimension)
        alpha = F.softmax(self.w_a(h), dim=-1)
        
        # Gated fusion
        h_prime = c * h + (1 - c) * alpha * h
        out = self.ffn(self.layer_norm(h_prime))
        return out

class DRCFNet(nn.Module):
    def __init__(self, dim_v=35, dim_a=74, dim_t=300, d=128, n_heads=4, dropout=0.2, num_layers=3):
        super(DRCFNet, self).__init__()
        self.d = d
        
        # Step 1: Multi-Scale Feature Projection
        self.proj_v = MSTCNProjection(dim_v, d, dropout)
        self.proj_a = MSTCNProjection(dim_a, d, dropout)
        self.proj_t = MSTCNProjection(dim_t, d, dropout)
        
        # Step 2: Temporal Transformer
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_v = nn.TransformerEncoder(encoder_layer_v, num_layers=num_layers)
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_a = nn.TransformerEncoder(encoder_layer_a, num_layers=num_layers)
        
        encoder_layer_t = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_t = nn.TransformerEncoder(encoder_layer_t, num_layers=num_layers)
        
        # Step 3: Gated MSR/SSR Split
        self.split_v = GatedSplit(d)
        self.split_a = GatedSplit(d)
        self.split_t = GatedSplit(d)
        
        # Step 4: CRE (Cross-modal Representation Encoder)
        self.cre_ta = nn.MultiheadAttention(embed_dim=d//2, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cre_tv = nn.MultiheadAttention(embed_dim=d//2, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # Step 5: Multi-Head Graph Neural Fusion
        self.gnn = MultiHeadGraphFusion(d_model=d//2, n_heads=n_heads)
        
        # Step 6: Gated Controlled Fusion (per node)
        self.gcf_t = GCFModule(d_model=d//2)
        self.gcf_a = GCFModule(d_model=d//2)
        self.gcf_v = GCFModule(d_model=d//2)
        self.gcf_ta = GCFModule(d_model=d//2)
        self.gcf_tv = GCFModule(d_model=d//2)
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.empty(1, 100, d))
        nn.init.uniform_(self.pos_emb, -0.01, 0.01)
        
        # Step 7: Final Prediction
        concat_dim = 5 * (d // 2)
        self.pool_attn = nn.Linear(concat_dim, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def forward(self, vision, audio, text):
        # Step 1: MS-TCN Projection
        v = self.proj_v(vision.permute(0, 2, 1))
        a = self.proj_a(audio.permute(0, 2, 1))
        t = self.proj_t(text.permute(0, 2, 1))
        
        # Add Positional Encoding
        v = v + self.pos_emb[:, :v.size(1), :]
        a = a + self.pos_emb[:, :a.size(1), :]
        t = t + self.pos_emb[:, :t.size(1), :]
        
        # Step 2: Temporal Transformer
        h_v = self.transformer_v(v)
        h_a = self.transformer_a(a)
        h_t = self.transformer_t(t)
        
        # Step 3: Gated Split
        msr_v, ssr_v = self.split_v(h_v)
        msr_a, ssr_a = self.split_a(h_a)
        msr_t, ssr_t = self.split_t(h_t)
        
        # Step 4: CRE
        z_ta_attn, _ = self.cre_ta(query=ssr_t, key=ssr_a, value=ssr_a)
        z_tv_attn, _ = self.cre_tv(query=ssr_t, key=ssr_v, value=ssr_v)
        
        z_ta = F.layer_norm(z_ta_attn + ssr_t, z_ta_attn.shape[-1:])
        z_tv = F.layer_norm(z_tv_attn + ssr_t, z_tv_attn.shape[-1:])
        
        # Step 5: MH-Graph Fusion
        gnn_nodes = [msr_t, msr_a, msr_v, z_ta, z_tv]
        h1, h2, h3, h4, h5 = self.gnn(gnn_nodes)
        
        # Step 6: GCF
        g1 = self.gcf_t(h1)
        g2 = self.gcf_a(h2)
        g3 = self.gcf_v(h3)
        g4 = self.gcf_ta(h4)
        g5 = self.gcf_tv(h5)
        
        # Step 7: Predict
        concatenated = torch.cat([g1, g2, g3, g4, g5], dim=-1)
        pool_weights = F.softmax(self.pool_attn(concatenated), dim=1)
        pooled = (pool_weights * concatenated).sum(dim=1)
        output = self.fc(pooled)
        
        return output, {
            'msr_v': msr_v, 'ssr_v': ssr_v,
            'msr_a': msr_a, 'ssr_a': ssr_a,
            'msr_t': msr_t, 'ssr_t': ssr_t
        }
