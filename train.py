import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping


def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, grad_clip, early_stopping):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for seq, struct, target in train_loader:
            optimizer.zero_grad()
            outputs = model(seq, struct).squeeze()
            target = target.squeeze()  # Ensure target is also squeezed to match outputs
            loss = criterion(outputs, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break


def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for seq, struct, target in val_loader:
            outputs = model(seq, struct).squeeze()
            target = target.squeeze()  # Ensure target is also squeezed to match outputs
            loss = criterion(outputs, target)
            val_loss += loss.item()
            all_outputs.extend(outputs.view(-1).cpu().numpy())  # Ensure outputs are correctly shaped
            all_targets.extend(target.view(-1).cpu().numpy())  # Ensure targets are correctly shaped

    val_loss /= len(val_loader)
    rmse = np.sqrt(((np.array(all_outputs) - np.array(all_targets)) ** 2).mean())
    mae = np.mean(np.abs(np.array(all_outputs) - np.array(all_targets)))

    return val_loss, rmse, mae


def cross_validate_model(model_func, dataset, k, epochs, batch_size, learning_rate, grad_clip):
    lengths = [len(dataset) // k for _ in range(k)]
    lengths[-1] += len(dataset) - sum(lengths)  # Adjust the last split to cover any remainder
    kfold = torch.utils.data.random_split(dataset, lengths)
    val_losses, rmses, maes = [], [], []

    for i, val_set in enumerate(kfold):
        train_sets = [kfold[j] for j in range(k) if j != i]
        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        model = model_func()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        train_model(model, train_loader, criterion, optimizer, scheduler, epochs, grad_clip, early_stopping)
        val_loss, rmse, mae = evaluate_model(model, val_loader, criterion)

        val_losses.append(val_loss)
        rmses.append(rmse)
        maes.append(mae)

    avg_val_loss = np.mean(val_losses)
    avg_rmse = np.mean(rmses)
    avg_mae = np.mean(maes)

    return avg_val_loss, avg_rmse, avg_mae

















