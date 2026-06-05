import torch
from tqdm import tqdm
import numpy as np
import copy

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cpu"):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.criterion      = criterion
        self.lr_scheduler   = scheduler
        self.device         = device

    def fit(self, train_loader, val_loader, epochs):
        loss_list = []                                              # Store average training loss per epoch
        accuracy_list = []                                          # Store validation accuracy per epoch
        accuracy_best = 0                                           # Keep track of the highest validation accuracy
        best_model_wts = copy.deepcopy(self.model.state_dict())     # Backup best model weights

        for epoch in tqdm(range(epochs)):
            epoch_loss      = self._train_one_epoch(train_loader)
            epoch_val_acc   = self._validate(val_loader)

            # Store epoch results
            loss_list.append(epoch_loss)
            accuracy_list.append(epoch_val_acc)

            # Adjust learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Save best model
            if epoch_val_acc > accuracy_best:
                accuracy_best = epoch_val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print(f"Epoch {epoch+1} done")
            print(f"Loss: {epoch_loss:.4f} | Val. Accuracy: {epoch_val_acc:.4f}\n")

        print("Training finished")
        print(f"Best accuracy: {accuracy_best:.4f}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        return loss_list, accuracy_list, self.model

    def _train_one_epoch(self, train_loader):
        batch_loss = []                                     # List to store batch's loss
        self.model.train()                                  # Set model to training mode

        # Training phase
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)     # Move tensors to device
            z = self.model(x)                               # Forward pass
            loss = self.criterion(z, y)                     # Compute loss
            batch_loss.append(loss.item())                  # Store individual batch losses for this epoch
            loss.backward()                                 # Backpropagation
            self.optimizer.step()                           # Update weights
            self.optimizer.zero_grad()                      # Reset gradients

        return np.mean(batch_loss)                          # Return mean batch loss

    def _validate(self, validation_loader):
        correct = 0                                         # Store correctly predicted samples
        total_val_samples = len(validation_loader.dataset)  # Calculate total validation samples
        self.model.eval()                                   # Set model to evaluation mode

        with torch.no_grad():
            for x_test, y_test in validation_loader:
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                z = self.model(x_test)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == y_test).sum().item()

        return correct / total_val_samples