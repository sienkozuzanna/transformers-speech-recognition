from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:        
        model: The neural network model to train.
        loader: DataLoader providing the training data.
        optimizer: The optimizer for updating model parameters.
        criterion: The loss function to optimize.
        device: The device to perform computations on.

    Returns:        
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, n = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:        
        model: The neural network model to evaluate.
        loader: DataLoader providing the validation data.
        criterion: The loss function to compute validation loss.
        device: The device to perform computations on.

    Returns:
        Average loss, accuracy, predictions, and true labels for the validation set.
    """
    model.eval()
    total_loss, correct, n = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average='macro')
    return total_loss / n, correct / n, macro_f1, preds, labels

def train(model, train_ds, valid_ds, epochs=30, batch_size=64, lr=3e-4, device='cuda', checkpoint_name=None, verbose=True, verbose_interval=1):
    """
    Trains the model using the provided training and validation datasets.
    
    Args:
        model: The neural network model to train.
        train_ds: The training dataset.
        valid_ds: The validation dataset.
        epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        lr: Learning rate for the optimizer.
        device: Device to perform training on.
        checkpoint_name: Optional name for saving model checkpoints. If None, checkpoints will not be saved.
        verbose: If True, prints training progress for each epoch.
        verbose_interval: Interval (in epochs) at which to print training progress when verbose is True.
    """
    if device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'valid_f1': []}
    best_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc, valid_f1, preds, labels = eval_epoch(model, valid_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['valid_f1'].append(valid_f1)

        if verbose and (epoch + 1) % verbose_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | "
                  f"Valid Acc: {valid_acc:.4f} | "
                  f"Valid Macro F1: {valid_f1:.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            if checkpoint_name:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics_history': history,
                    'valid_acc': valid_acc,
                }, f'{checkpoint_name}_best.pth')
                print(f"  -> saved (val_acc={best_acc:.4f})")

    if checkpoint_name:
        torch.save(history, f'{checkpoint_name}_history.pth')

    print(f"Best validation accuracy: {best_acc:.4f}")
    return model, history

def predict(model, test_ds, device='cuda', batch_size=64):
    """
    Generates predictions for the test dataset using the trained model.

    Args:
        model: The trained neural network model.
        test_ds: The test dataset for which to generate predictions.
        device: Device to perform inference on.
        batch_size: Batch size for generating predictions.  
    Returns:
        A tuple containing the predicted class labels and the true labels for the test dataset.
    """

    if device == 'cuda':
        pin_memory = True
    else:
        pin_memory=False
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return preds, labels