import torch
from tqdm import tqdm

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device):

    model.to(device)
    criterion.to(device)

    scaler = torch.cuda.amp.GradScaler()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_valid_loss = float('inf')
    best_model_state = None
    patience, counter = 5, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in loop:
            vision, audio, text, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds, components = model(vision, audio, text)
                loss, _ = criterion(preds, labels, components)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        val_loss, *_ = test(model, valid_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Valid: {val_loss:.4f}")

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def compute_metrics(preds, labels):
    mae = torch.mean(torch.abs(preds - labels))
    
    # Binary accuracy
    preds_bin = (preds > 0).float()
    labels_bin = (labels > 0).float()
    
    acc = (preds_bin == labels_bin).float().mean()
    
    return mae.item(), acc.item()
    
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
