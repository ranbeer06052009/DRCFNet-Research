import os
import torch
import torch.nn.functional as F
from evaluation.performance import eval_affect

def hsic(h1, h2):
    """Hilbert-Schmidt Independence Criterion"""
    n = h1.size(0)
    if n < 2: return torch.tensor(0.0, device=h1.device)
    
    k1 = torch.matmul(h1, h1.t())
    k2 = torch.matmul(h2, h2.t())
    
    H = torch.eye(n, device=h1.device) - (1.0/n) * torch.ones(n, n, device=h1.device)
    return (1.0 / ((n-1)**2)) * torch.trace(torch.matmul(torch.matmul(H, k1), torch.matmul(H, k2)))

def mea_criterion(pred, truth, features, model, task_criterion, alpha=2e-2, beta=3e-2):
    """
    Computes total MEA loss: Task Loss + Disparity Loss (HSIC) + Adversarial Losses
    """
    l_task = task_criterion(pred.squeeze(), truth.squeeze())
    
    h_e = features['h_e'] # [L, V, A]
    h_a = features['h_a'] # [L, V, A]
    
    # 1. HSIC Disparity Loss
    l_dis = 0.0
    for i in range(3):
        l_dis += hsic(h_e[i], h_a[i])
    l_dis = l_dis / 3.0
    
    # 2. Adversarial Losses
    imp_logits, mod_e_logits, mod_a_logits = model.compute_discriminator_logits(features)
    
    l_agn = 0.0
    l_exc = 0.0
    
    for m in range(3):
        # Ground truth labels for modality
        y_m = torch.full((h_e[m].size(0),), m, dtype=torch.long, device=pred.device)
        
        # Importance Discriminator output
        imp_probs = F.softmax(imp_logits[m], dim=-1)
        # Probability of being correctly identified as modality m
        prob_m = imp_probs[torch.arange(imp_probs.size(0)), m]
        omega = (1.0 - prob_m).detach() # Regulatory factor
        
        # Modality Discriminator on agnostic features (GRL is applied in compute_discriminator_logits)
        # Cross entropy computes -log(P(y_m))
        ce_agn = F.cross_entropy(mod_a_logits[m], y_m, reduction='none')
        l_agn += torch.mean(omega * ce_agn)
        
        # Modality Discriminator on exclusive features
        l_exc += F.cross_entropy(mod_e_logits[m], y_m)
        
    l_agn = l_agn / 3.0
    l_exc = l_exc / 3.0
    
    # Total loss
    loss = l_task + alpha * l_dis + beta * (l_agn + l_exc)
    
    return loss, l_task, l_dis, l_agn, l_exc

def train_epoch_mea(model, dataloader, optimizer, task_criterion, device='cuda', clip_grad=1.0, alpha=2e-2, beta=3e-2):
    model.train()
    total_loss, total_task, total_dis, total_agn, total_exc = 0.0, 0.0, 0.0, 0.0, 0.0
    
    for batch in dataloader:
        vision, audio, text, labels = batch
        vision = vision.to(device)
        audio = audio.to(device)
        text = text.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, features = model(vision, audio, text, kg_features=None)
        
        loss, l_task, l_dis, l_agn, l_exc = mea_criterion(logits, labels, features, model, task_criterion, alpha, beta)
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_task += l_task.item()
        total_dis += l_dis.item()
        total_agn += l_agn.item()
        total_exc += l_exc.item()
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_task / num_batches, total_dis / num_batches, total_agn / num_batches, total_exc / num_batches

def test_mea(model, dataloader, task_criterion, device='cuda', alpha=2e-2, beta=3e-2, return_preds=False):
    model.eval()
    total_loss, total_task = 0.0, 0.0
    
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch in dataloader:
            vision, audio, text, labels = batch
            vision = vision.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)
            
            logits, features = model(vision, audio, text, kg_features=None)
            
            loss, l_task, _, _, _ = mea_criterion(logits, labels, features, model, task_criterion, alpha, beta)
            
            total_loss += loss.item()
            total_task += l_task.item()
            
            all_preds.append(logits.cpu())
            all_truths.append(labels.cpu())
            
    num_batches = len(dataloader)
    
    preds = torch.cat(all_preds, dim=0)
    truths = torch.cat(all_truths, dim=0)
    
    acc = eval_affect(truths, preds)
    
    metrics = {
        'Loss': total_loss / num_batches,
        'Task': total_task / num_batches,
        'Accuracy': acc
    }
    
    if return_preds:
        return metrics, preds, truths
    return metrics

def train_mea_loop(model, train_loader, valid_loader, task_criterion, optimizer, epochs=60, device='cuda', scheduler=None, clip_grad=1.0, alpha=2e-2, beta=3e-2, save_path='best_mea_model.pth'):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_task': [], 'train_dis': [], 'train_agn': [], 'train_exc': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        t_loss, t_task, t_dis, t_agn, t_exc = train_epoch_mea(model, train_loader, optimizer, task_criterion, device, clip_grad, alpha, beta)
        
        val_metrics = test_mea(model, valid_loader, task_criterion, device, alpha, beta)
        v_loss = val_metrics['Loss']
        v_acc = val_metrics['Accuracy']
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['train_task'].append(t_task)
        history['train_dis'].append(t_dis)
        history['train_agn'].append(t_agn)
        history['train_exc'].append(t_exc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Task: {t_task:.4f} | Dis: {t_dis:.4f} | Agn: {t_agn:.4f} | Exc: {t_exc:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
        
        if scheduler is not None:
            scheduler.step(v_loss)
            
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved best model with Val Loss: {v_loss:.4f}")
            
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model, history
