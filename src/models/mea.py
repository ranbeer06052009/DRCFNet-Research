import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class UnimodalExtractor(nn.Module):
    def __init__(self, in_dim, d, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, d, kernel_size, padding=kernel_size//2)
        self.pos_emb = PositionalEncoding(d)
        self.bilstm = nn.LSTM(d, d // 2, bidirectional=True, batch_first=True)
        
    def forward(self, x):
        # x: (B, T, in_dim)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.pos_emb(x)
        x, _ = self.bilstm(x)
        return x

class PSALayer(nn.Module):
    def __init__(self, d, n_heads=8, mu=0.25):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d // n_heads
        self.mu = mu
        
        self.ln1 = nn.LayerNorm(d)
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        
        self.cnn = nn.Conv2d(n_heads, n_heads, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        
        self.w_o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Linear(4*d, d)
        )
        self.ln3 = nn.LayerNorm(d)
        
    def forward(self, x, prev_attn=None):
        B, T, d = x.shape
        nx = self.ln1(x)
        
        q = self.w_q(nx).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(nx).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(nx).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, H, T, T)
        
        if prev_attn is not None:
            pred_attn = self.gelu(self.cnn(prev_attn)) # (B, H, T, T)
            attn = self.mu * pred_attn + (1 - self.mu) * F.softmax(logits, dim=-1)
            # Ensure it normalizes
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            pred_attn = self.gelu(self.cnn(logits)) # Predict for next layer
            attn = F.softmax(logits, dim=-1)
            
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, d)
        out = self.w_o(out)
        
        x = self.ln2(x + out)
        x = self.ln3(x + self.ffn(x))
        return x, pred_attn

class PSAModule(nn.Module):
    def __init__(self, d, num_layers=2, n_heads=8, mu=0.25):
        super().__init__()
        self.layers = nn.ModuleList([PSALayer(d, n_heads, mu) for _ in range(num_layers)])
        
    def forward(self, x):
        prev_attn = None
        for layer in self.layers:
            x, prev_attn = layer(x, prev_attn)
        return x

class WAL(nn.Module):
    def __init__(self, d, t_l, t_v, t_a):
        super().__init__()
        self.w_l = nn.Linear(t_l * d, 1)
        self.w_v = nn.Linear(t_v * d, 1)
        self.w_a = nn.Linear(t_a * d, 1)
        
    def forward(self, z_l, z_v, z_a):
        B = z_l.shape[0]
        
        # Flatten
        f_l = z_l.view(B, -1)
        f_v = z_v.view(B, -1)
        f_a = z_a.view(B, -1)
        
        g_l = self.w_l(f_l).tanh()
        g_v = self.w_v(f_v).tanh()
        g_a = self.w_a(f_a).tanh()
        
        weights = F.softmax(torch.cat([g_l, g_v, g_a], dim=1), dim=1) # (B, 3)
        
        w_l = weights[:, 0].view(B, 1, 1)
        w_v = weights[:, 1].view(B, 1, 1)
        w_a = weights[:, 2].view(B, 1, 1)
        
        return z_l * w_l, z_v * w_v, z_a * w_a

class MRU(nn.Module):
    def __init__(self, d, n_heads=8):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Linear(4*d, d)
        )
        self.ln2 = nn.LayerNorm(d)
        
    def forward(self, source, target):
        q = self.ln_q(target)
        kv = self.ln_kv(source)
        
        attn_out, _ = self.mha(q, kv, kv)
        out = self.ln1(target + attn_out)
        out = self.ln2(out + self.ffn(out))
        return out

class HCAModule(nn.Module):
    def __init__(self, d, n_heads=8, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # We need MRUs for L, V, A. For simplicity we encapsulate one HCA layer for all
        self.mru_mixed_l = MRU(d, n_heads)
        self.mru_mixed_v = MRU(d, n_heads)
        self.mru_mixed_a = MRU(d, n_heads)
        
        self.mru_coarse_l = MRU(d, n_heads)
        self.mru_coarse_v = MRU(d, n_heads)
        self.mru_coarse_a = MRU(d, n_heads)
        
        self.mru_fine_l1 = MRU(d, n_heads)
        self.mru_fine_l2 = MRU(d, n_heads)
        
        self.mru_fine_v1 = MRU(d, n_heads)
        self.mru_fine_v2 = MRU(d, n_heads)
        
        self.mru_fine_a1 = MRU(d, n_heads)
        self.mru_fine_a2 = MRU(d, n_heads)
        
    def forward(self, z_l, z_v, z_a):
        for _ in range(self.num_layers):
            z_lva = torch.cat([z_l, z_v, z_a], dim=1)
            z_la = torch.cat([z_l, z_a], dim=1)
            z_lv = torch.cat([z_l, z_v], dim=1)
            z_va = torch.cat([z_v, z_a], dim=1)
            
            # Target L
            z_l_mixed = self.mru_mixed_l(z_lva, z_l)
            z_l_coarse = self.mru_coarse_l(z_va, z_l_mixed)
            z_l = self.mru_fine_l1(z_v, z_l_coarse) + self.mru_fine_l2(z_a, z_l_coarse)
            
            # Target V
            z_v_mixed = self.mru_mixed_v(z_lva, z_v)
            z_v_coarse = self.mru_coarse_v(z_la, z_v_mixed)
            z_v = self.mru_fine_v1(z_l, z_v_coarse) + self.mru_fine_v2(z_a, z_v_coarse)
            
            # Target A
            z_a_mixed = self.mru_mixed_a(z_lva, z_a)
            z_a_coarse = self.mru_coarse_a(z_lv, z_a_mixed)
            z_a = self.mru_fine_a1(z_l, z_a_coarse) + self.mru_fine_a2(z_v, z_a_coarse)
            
        return z_l, z_v, z_a

class DecoupledEncoder(nn.Module):
    def __init__(self, d, dh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, dh),
            nn.GELU(),
            nn.Linear(dh, dh)
        )
    def forward(self, x):
        # average pooling before MLP
        x = x.mean(dim=1)
        return self.net(x)

class DGF(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.w_e = nn.Linear(dh, dh, bias=False)
        self.q = nn.Sequential(
            nn.Linear(2*dh, dh),
            nn.GELU(),
            nn.Linear(dh, 1)
        )
        
    def forward(self, h_l, h_v, h_a):
        nodes = [h_l, h_v, h_a] # List of (B, dh)
        proj_nodes = [self.w_e(n) for n in nodes]
        
        enhanced = []
        for i in range(3):
            scores = []
            for j in range(3):
                pair = torch.cat([proj_nodes[i], proj_nodes[j]], dim=1)
                score = F.gelu(self.q(pair))
                scores.append(score)
            scores = torch.cat(scores, dim=1) # (B, 3)
            attn = F.softmax(scores, dim=1).unsqueeze(2) # (B, 3, 1)
            
            agg = torch.stack(proj_nodes, dim=1) # (B, 3, dh)
            agg = (agg * attn).sum(dim=1) # (B, dh)
            enhanced.append(torch.sigmoid(agg))
            
        return sum(enhanced) # H_fin

class MEA(nn.Module):
    def __init__(self, dim_l=300, dim_v=35, dim_a=74, t_l=50, t_v=50, t_a=50, d=40, dh=64, n_heads=8, mu=0.25):
        super().__init__()
        self.d = d
        self.dh = dh
        
        # Unimodal Extractors
        self.ext_l = UnimodalExtractor(dim_l, d, kernel_size=3)
        self.ext_v = UnimodalExtractor(dim_v, d, kernel_size=3)
        self.ext_a = UnimodalExtractor(dim_a, d, kernel_size=3) # paper mentions 5 for audio sometimes but 3 in generic text
        
        # PSA Modules
        self.psa_l = PSAModule(d, n_heads=n_heads, mu=mu)
        self.psa_v = PSAModule(d, n_heads=n_heads, mu=mu)
        self.psa_a = PSAModule(d, n_heads=n_heads, mu=mu)
        
        # Weighted Attention Layer
        self.wal = WAL(d, t_l, t_v, t_a)
        
        # HCA Modules
        self.hca = HCAModule(d, n_heads=n_heads)
        
        # Encoders
        self.exc_enc_l = DecoupledEncoder(d, dh)
        self.exc_enc_v = DecoupledEncoder(d, dh)
        self.exc_enc_a = DecoupledEncoder(d, dh)
        
        self.agn_enc = DecoupledEncoder(d, dh)
        
        # Discriminators
        self.importance_disc = nn.Sequential(
            nn.Linear(dh, 3)
        )
        self.modality_disc = nn.Sequential(
            nn.Linear(dh, 3)
        )
        self.grl = GRL()
        
        # Decoupled Graph Fusion
        self.dgf_het = DGF(dh)
        self.dgf_hom = DGF(dh)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dh * 2, dh),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dh, 1)
        )
        
    def forward(self, vision, audio, text, kg_features=None):
        # 1. Unimodal Extraction
        z_l = self.ext_l(text)
        z_v = self.ext_v(vision)
        z_a = self.ext_a(audio)
        
        # 2. PSA
        z_l_e = self.psa_l(z_l)
        z_v_e = self.psa_v(z_v)
        z_a_e = self.psa_a(z_a)
        
        # WAL
        z_l_e, z_v_e, z_a_e = self.wal(z_l_e, z_v_e, z_a_e)
        
        # 3. HCA
        z_l_a, z_v_a, z_a_a = self.hca(z_l, z_v, z_a)
        
        # 4. Decoupled Representations
        h_l_e = self.exc_enc_l(z_l_e)
        h_v_e = self.exc_enc_v(z_v_e)
        h_a_e = self.exc_enc_a(z_a_e)
        
        h_l_a = self.agn_enc(z_l_a)
        h_v_a = self.agn_enc(z_v_a)
        h_a_a = self.agn_enc(z_a_a)
        
        # 5. Fusion
        h_fin_e = self.dgf_het(h_l_e, h_v_e, h_a_e)
        h_fin_a = self.dgf_hom(h_l_a, h_v_a, h_a_a)
        
        h_fin = torch.cat([h_fin_e, h_fin_a], dim=1)
        
        # 6. Prediction
        out = self.classifier(h_fin)
        
        # 7. Outputs for Discriminators / Loss calculation
        features = {
            'h_e': [h_l_e, h_v_e, h_a_e],
            'h_a': [h_l_a, h_v_a, h_a_a]
        }
        
        return out, features

    def compute_discriminator_logits(self, features):
        h_e = features['h_e']
        h_a = features['h_a']
        
        # Importance Discriminator (only on h_a)
        imp_logits = [self.importance_disc(h) for h in h_a]
        
        # Modality Discriminator on h_e (direct)
        mod_e_logits = [self.modality_disc(h) for h in h_e]
        
        # Modality Discriminator on h_a (with GRL)
        mod_a_logits = [self.modality_disc(self.grl(h)) for h in h_a]
        
        return imp_logits, mod_e_logits, mod_a_logits
