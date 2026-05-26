import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import pandas as pd

class Trainer:
    def __init__(self, model, optimizer, device, save_path):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for images, labels, in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    
    def evaluate(self, loader):
        self.model.eval()
        
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)

                predictions = outputs.argmax(dim=1).cpu()

                all_predictions = torch.cat(all_predictions).numpy()
                all_labels = torch.cat(all_labels).numpy()

            f1 = f1_score(all_labels, all_predictions, average="macro")
            
            return f1
        
    def train(self, train_loader, val_loader, epochs):
        best_f1 = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_f1 = self.evaluate(val_loader)

            print(f'Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state.dict(), self.save_path)
                print(f'Saved best model (F1: {best_f1:.4f})')

        return best_f1