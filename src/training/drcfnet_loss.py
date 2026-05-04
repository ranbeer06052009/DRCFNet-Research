import torch
import torch.nn as nn
import torch.nn.functional as F

class DRCFNetLoss(nn.Module):
    def __init__(self, lambda_orth=0.01, lambda_contrastive=0.01, task_weight=1.0, temperature=0.07):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_contrastive = lambda_contrastive
        self.task_weight = task_weight
        self.temperature = temperature
        self.task_loss_fn = nn.L1Loss()
        
    def forward(self, preds, labels, components):
        
        # ===== 1. TASK LOSS =====
        l_task = self.task_loss_fn(preds, labels)
        
        # ===== 2. ORTHOGONALITY LOSS =====
        l_orth = 0.0
        for m in ['v','a','t']:
            msr = components[f'msr_{m}']
            ssr = components[f'ssr_{m}']
            
            msr = F.normalize(msr, dim=-1)
            ssr = F.normalize(ssr, dim=-1)
            
            orth = torch.bmm(msr.transpose(1,2), ssr)
            l_orth += (orth ** 2).mean()
        
        l_orth = l_orth / 3
        
        # ===== 3. CONTRASTIVE LOSS =====
        ssr_t = components['ssr_t'].mean(dim=1)
        ssr_a = components['ssr_a'].mean(dim=1)
        ssr_v = components['ssr_v'].mean(dim=1)
        
        ssr_t = F.normalize(ssr_t, dim=-1)
        ssr_a = F.normalize(ssr_a, dim=-1)
        ssr_v = F.normalize(ssr_v, dim=-1)
        
        labels_idx = torch.arange(ssr_t.size(0)).to(ssr_t.device)
        
        sim_ta = torch.matmul(ssr_t, ssr_a.T)
        sim_tv = torch.matmul(ssr_t, ssr_v.T)
        
        l_ta = F.cross_entropy(sim_ta / self.temperature, labels_idx)
        l_tv = F.cross_entropy(sim_tv / self.temperature, labels_idx)
        
        l_contrastive = (l_ta + l_tv) / 2
        
        # ===== TOTAL =====
        total_loss = (
            self.task_weight * l_task
            + self.lambda_orth * l_orth
            + self.lambda_contrastive * l_contrastive
        )
        
        return total_loss, {
            'task_loss': l_task.item(),
            'orth_loss': l_orth.item(),
            'contrastive_loss': l_contrastive.item()
        }