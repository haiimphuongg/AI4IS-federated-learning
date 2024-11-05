'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report


int_to_labels = {0:'airplane',1: 'automobile',2: 'bird',3: 'cat',4: 'deer',5: 'dog',6: 'frog',7: 'horse',8: 'ship',9: 'truck'}


def xavier_init(in_features, out_features, weight):
    x = sqrt(6/(in_features + out_features))
    print(x)
    weight.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-x, x))
    return weight


class Classifier(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear = xavier_init(in_features, out_features, nn.Linear(in_features, out_features))
    def forward(self, x):
        return self.linear(x)
    def predict(self, x):
        logits = self.forward(x)
        return logits, torch.max(logits.data, 1)[1]
    def predict_label(self, target):
        return int_to_labels[target]
    

def validate(model, val_dataloader, criterion = CrossEntropyLoss(), device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        running_loss = 0
        for images, labels in tqdm(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs, predictions = model.predict(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy()) 
        
        avg_val_loss = running_loss/len(val_dataloader)
        return avg_val_loss, all_labels, all_predictions


def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, show_step = 1, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    for epoch in range(num_epochs):
        model.train()
        print("Training epoch ", epoch + 1)
        
        running_loss = 0
        
        for images, labels in tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_epoch_train_loss = running_loss/len(train_dataloader)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_train_loss:.4f}')
        avg_val_loss, all_val_labels, all_val_predictions = validate(model, val_dataloader, criterion)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        if (epoch%show_step == 0 or epoch == num_epochs - 1):
            print(classification_report(all_val_labels, all_val_predictions, zero_division = True))
        scheduler.step(avg_val_loss)
        # print("Learning rate: ", scheduler.get_last_lr()[0])

        weights = model.state_dict()
        return avg_epoch_train_loss, weights
    
'''
Usage:
    
    python server.py
    
'''