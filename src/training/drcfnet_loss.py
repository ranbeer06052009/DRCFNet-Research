import torch
import torch.nn as nn
import torch.nn.functional as F

class DRCFNetLoss(nn.Module):
    def __init__(self, lambda_orth=0.01, lambda_contrastive=0.01, task_weight=1.0):
        super(DRCFNetLoss, self).__init__()
        self.lambda_orth = lambda_orth
        self.lambda_contrastive = lambda_contrastive
        self.task_weight = task_weight
        # L1 Loss (MAE) is standard for MOSI/MOSEI sentiment regression
        self.task_loss_fn = nn.L1Loss()
        
    def forward(self, preds, labels, components):
        # 1. Task Loss
        l_task = self.task_loss_fn(preds, labels)
        
        # 2. Orthogonality Loss
        # L_orth = sum(||(MSR_m)^T SSR_m||_F^2)
        l_orth = 0.0
        for m in ['v', 'a', 't']:
            msr = components[f'msr_{m}']
            ssr = components[f'ssr_{m}']
            # Batch matrix multiplication: (Batch, Dim, Seq) x (Batch, Seq, Dim) -> (Batch, Dim, Dim)
            orth_matrix = torch.bmm(msr.transpose(1, 2), ssr)
            # Frobenius norm squared, averaged over batch
            l_orth += (orth_matrix ** 2).mean()
            
        # 3. Contrastive Alignment Loss
        # Pull SSRs closer together: MSE(ssr_t, ssr_a) + MSE(ssr_t, ssr_v)
        ssr_t = components['ssr_t']
        ssr_a = components['ssr_a']
        ssr_v = components['ssr_v']
        
        l_contrastive = F.mse_loss(ssr_t, ssr_a) + F.mse_loss(ssr_t, ssr_v)
        
        # Total Loss
        total_loss = self.task_weight * l_task + self.lambda_orth * l_orth + self.lambda_contrastive * l_contrastive
        
        return total_loss, {
            'task_loss': l_task.item(),
            'orth_loss': l_orth.item() if isinstance(l_orth, torch.Tensor) else l_orth,
            'contrastive_loss': l_contrastive.item()
        }
