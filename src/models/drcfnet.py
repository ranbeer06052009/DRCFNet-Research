import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNeuralFusion(nn.Module):
    def __init__(self, d_model):
        super(GraphNeuralFusion, self).__init__()
        self.w_h = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, nodes):
        # nodes is a list of tensors, each of shape (B, T, d_model)
        stacked_nodes = torch.stack(nodes, dim=2) # (B, T, N, d_model)
        B, T, N, d = stacked_nodes.size()
        
        # Step 1: Projection
        tilde_h = self.w_h(stacked_nodes) # (B, T, N, d)
        
        # Step 2: Relation score (Vectorized)
        tilde_i = tilde_h.unsqueeze(3) # (B, T, N, 1, d)
        tilde_j = tilde_h.unsqueeze(2) # (B, T, 1, N, d)
        
        pair = torch.cat([
            tilde_i.expand(-1, -1, -1, N, -1),
            tilde_j.expand(-1, -1, N, -1, -1)
        ], dim=-1) # (B, T, N, N, 2d)
        
        delta = self.ffn(pair).squeeze(-1) # (B, T, N, N)
                
        # Step 3: Attention weight
        xi = F.softmax(delta, dim=-1) # (B, T, N, N)
        
        # Step 4: Message passing / Aggregation
        # h_new_i = sum_j xi_ij * tilde_h_j
        h_new = torch.sum(xi.unsqueeze(-1) * tilde_h.unsqueeze(2), dim=3) # (B, T, N, d)
        
        # Step 5: Final Output
        h_final = self.layer_norm(stacked_nodes + h_new) # (B, T, N, d)
        
        # Split back to list
        return [h_final[:, :, i, :] for i in range(N)]


class GCFModule(nn.Module):
    def __init__(self, d_model):
        super(GCFModule, self).__init__()
        self.w_c = nn.Linear(d_model, d_model)
        self.w_a = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, h):
        c = torch.sigmoid(self.w_c(h))
        alpha = torch.sigmoid(self.w_a(h))
        
        h_prime = c * h + (1 - c) * alpha * h
        out = self.ffn(self.layer_norm(h_prime))
        return out


class DRCFNet(nn.Module):
    def __init__(self, dim_v=35, dim_a=74, dim_t=300, d=128, n_heads=4, dropout=0.2, num_layers=3):
        super(DRCFNet, self).__init__()
        self.d = d
        
        # Step 1: Feature Projection with Temporal Context
        self.proj_v = nn.Sequential(nn.Conv1d(dim_v, d, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.proj_a = nn.Sequential(nn.Conv1d(dim_a, d, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.proj_t = nn.Sequential(nn.Conv1d(dim_t, d, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        
        self.ln_v = nn.LayerNorm(d)
        self.ln_a = nn.LayerNorm(d)
        self.ln_t = nn.LayerNorm(d)
        
        # Step 2: Temporal Transformer
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_v = nn.TransformerEncoder(encoder_layer_v, num_layers=num_layers)
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_a = nn.TransformerEncoder(encoder_layer_a, num_layers=num_layers)
        
        encoder_layer_t = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_t = nn.TransformerEncoder(encoder_layer_t, num_layers=num_layers)
        
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
        
        # Step 5: Graph Neural Fusion
        self.gnn = GraphNeuralFusion(d_model=d//2)
        
        # Step 6: Gated Controlled Fusion (per node)
        self.gcf_t = GCFModule(d_model=d//2)
        self.gcf_a = GCFModule(d_model=d//2)
        self.gcf_v = GCFModule(d_model=d//2)
        self.gcf_ta = GCFModule(d_model=d//2)
        self.gcf_tv = GCFModule(d_model=d//2)
        
        # Positional Encoding (Max sequence length set to 100 to accommodate both MOSI and MOSEI)
        self.pos_emb = nn.Parameter(torch.empty(1, 100, d))
        nn.init.uniform_(self.pos_emb, -0.01, 0.01)
        
        # Step 7: Final Prediction
        # Concatenation of 5 nodes = 5 * (d/2) = 2.5 * d
        concat_dim = 5 * (d // 2)
        
        self.pool_attn = nn.Linear(concat_dim, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(concat_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1) # Regression for sentiment
        )

    def forward(self, vision, audio, text):
        # Input shapes: (Batch, Seq_Len, Dim)
        # Conv1d expects (Batch, Channels, Length), so we permute, then permute back and normalize
        v = self.ln_v(self.proj_v(vision.permute(0, 2, 1)).permute(0, 2, 1))
        a = self.ln_a(self.proj_a(audio.permute(0, 2, 1)).permute(0, 2, 1))
        t = self.ln_t(self.proj_t(text.permute(0, 2, 1)).permute(0, 2, 1))
        
        # Add Positional Encoding
        v = v + self.pos_emb[:, :v.size(1), :]
        a = a + self.pos_emb[:, :a.size(1), :]
        t = t + self.pos_emb[:, :t.size(1), :]
        
        # Step 2
        h_v = self.transformer_v(v)
        h_a = self.transformer_a(a)
        h_t = self.transformer_t(t)
        
        # Step 3
        msr_v, ssr_v = self.w_ex_v(h_v), self.w_ag_v(h_v)
        msr_a, ssr_a = self.w_ex_a(h_a), self.w_ag_a(h_a)
        msr_t, ssr_t = self.w_ex_t(h_t), self.w_ag_t(h_t)
        
        # Step 4
        z_ta_attn, _ = self.cre_ta(query=ssr_t, key=ssr_a, value=ssr_a)
        z_tv_attn, _ = self.cre_tv(query=ssr_t, key=ssr_v, value=ssr_v)
        
        # Add Residual + LayerNorm
        z_ta = F.layer_norm(z_ta_attn + ssr_t, z_ta_attn.shape[-1:])
        z_tv = F.layer_norm(z_tv_attn + ssr_t, z_tv_attn.shape[-1:])
        
        # Step 5
        gnn_nodes = [msr_t, msr_a, msr_v, z_ta, z_tv]
        h1, h2, h3, h4, h5 = self.gnn(gnn_nodes)
        
        # Step 6
        g1 = self.gcf_t(h1)
        g2 = self.gcf_a(h2)
        g3 = self.gcf_v(h3)
        g4 = self.gcf_ta(h4)
        g5 = self.gcf_tv(h5)
        
        # Step 7
        concatenated = torch.cat([g1, g2, g3, g4, g5], dim=-1)
        
        # Attention-based Pooling over time
        pool_weights = F.softmax(self.pool_attn(concatenated), dim=1)
        pooled = (pool_weights * concatenated).sum(dim=1)
        output = self.fc(pooled)
        
        # Return output and necessary components for loss
        return output, {
            'msr_v': msr_v, 'ssr_v': ssr_v,
            'msr_a': msr_a, 'ssr_a': ssr_a,
            'msr_t': msr_t, 'ssr_t': ssr_t
        }
