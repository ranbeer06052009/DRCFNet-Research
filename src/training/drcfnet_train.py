import torch
from tqdm import tqdm

def train(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    epochs,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)
    criterion.to(device)
    
    best_valid_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        task_l, orth_l, contr_l = 0.0, 0.0, 0.0
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            # batch is (vision, audio, text, labels)
            vision = batch[0].to(device)
            audio = batch[1].to(device)
            text = batch[2].to(device)
            labels = batch[3].to(device)
            
            optimizer.zero_grad()
            
            preds, components = model(vision, audio, text)
            loss, loss_dict = criterion(preds, labels, components)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            task_l += loss_dict['task_loss']
            orth_l += loss_dict['orth_loss']
            contr_l += loss_dict['contrastive_loss']
            
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)
        task_l /= len(train_loader)
        orth_l /= len(train_loader)
        contr_l /= len(train_loader)
        
        # Validation
        val_loss, val_task, val_orth, val_contr = test(model, valid_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} (Task:{task_l:.4f} Orth:{orth_l:.4f} Contr:{contr_l:.4f})")
        print(f"Epoch {epoch+1} | Valid Loss: {val_loss:.4f} (Task:{val_task:.4f} Orth:{val_orth:.4f} Contr:{val_contr:.4f})")
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_model_state = model.state_dict()
            
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def test(model, dataloader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu', return_preds=False):
    model.eval()
    test_loss = 0.0
    task_l, orth_l, contr_l = 0.0, 0.0, 0.0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            vision = batch[0].to(device)
            audio = batch[1].to(device)
            text = batch[2].to(device)
            labels = batch[3].to(device)
            
            preds, components = model(vision, audio, text)
            loss, loss_dict = criterion(preds, labels, components)
            
            test_loss += loss.item()
            task_l += loss_dict['task_loss']
            orth_l += loss_dict['orth_loss']
            contr_l += loss_dict['contrastive_loss']
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    test_loss /= len(dataloader)
    task_l /= len(dataloader)
    orth_l /= len(dataloader)
    contr_l /= len(dataloader)
    
    if return_preds:
        return test_loss, torch.cat(all_preds), torch.cat(all_labels)
    return test_loss, task_l, orth_l, contr_l
