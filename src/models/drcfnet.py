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
        
        # Step 2: Relation score
        delta = torch.zeros(B, T, N, N, device=stacked_nodes.device)
        for i in range(N):
            for j in range(N):
                concat_h = torch.cat([tilde_h[:, :, i, :], tilde_h[:, :, j, :]], dim=-1) # (B, T, 2d)
                delta[:, :, i, j] = self.ffn(concat_h).squeeze(-1) # (B, T)
                
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
        alpha = F.softmax(self.w_a(h), dim=-1)
        
        h_prime = c * h + alpha * h
        out = self.ffn(self.layer_norm(h_prime))
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
        
        # Step 5: Graph Neural Fusion
        self.gnn = GraphNeuralFusion(d_model=d//2)
        
        # Step 6: Gated Controlled Fusion (per node)
        self.gcf_t = GCFModule(d_model=d//2)
        self.gcf_a = GCFModule(d_model=d//2)
        self.gcf_v = GCFModule(d_model=d//2)
        self.gcf_ta = GCFModule(d_model=d//2)
        self.gcf_tv = GCFModule(d_model=d//2)
        
        # Step 7: Final Prediction
        # Concatenation of 5 nodes = 5 * (d/2) = 2.5 * d
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
        
        # Global Average Pooling over time
        pooled = concatenated.mean(dim=1)
        output = self.fc(pooled)
        
        # Return output and necessary components for loss
        return output, {
            'msr_v': msr_v, 'ssr_v': ssr_v,
            'msr_a': msr_a, 'ssr_a': ssr_a,
            'msr_t': msr_t, 'ssr_t': ssr_t
        }
