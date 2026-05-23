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
