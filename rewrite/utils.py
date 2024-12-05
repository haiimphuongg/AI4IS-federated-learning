from collections import OrderedDict
import torch
import torch.nn as nn
import pandas as pd 
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from math import sqrt

int_to_labels = {0:'airplane',1: 'automobile',2: 'bird',3: 'cat',4: 'deer',5: 'dog',6: 'frog',7: 'horse',8: 'ship',9: 'truck'}


def xavier_init(in_features, out_features, weight):
    x = sqrt(6/(in_features + out_features))
    print(x)
    weight.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-x, x))
    return weight


class SoftmaxRegression(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)
    def predict(self, x):
        logits = self.forward(x)
        return logits, torch.max(logits.data, 1)[1]
    def predict_label(self, target):
        return int_to_labels[target]
    

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        embedding = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label
        
def train(model, train_dataloader, val_dataloader,
          num_epoch = 10, device = 'cuda' if torch.cuda.is_available() else 'cpu',
          lr = 0.01, weight_decay = 0.001, show_step = 1):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay = weight_decay, betas=(0.9, 0.999))

    for epoch in range(num_epoch):
        model.train()
        print("Training epoch ", epoch + 1)
        
        running_loss = 0
        train_predictions = []
        train_labels = []
        for embeddings, labels in tqdm(train_dataloader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_predictions.extend(torch.argmax(outputs, 1).tolist())
            train_labels.extend(labels.tolist())
            running_loss += loss.item()

        avg_epoch_train_loss = running_loss/len(train_dataloader)
        avg_val_loss, all_val_labels, all_val_predictions = validate(model, val_dataloader)
        train_accuracy = accuracy_score(train_predictions, train_labels)
        val_accuracy = accuracy_score(all_val_predictions, all_val_labels)
        train_f1 = f1_score(train_predictions, train_labels, average = 'macro')
        val_f1 = f1_score(all_val_predictions, all_val_labels, average = 'macro')
        
        
        print(f'Epoch [{epoch + 1}/{num_epoch}], Training Loss: {avg_epoch_train_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epoch}], Train Accuracy: {train_accuracy:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epoch}], Validation Loss: {avg_val_loss:.4f}') 
        print(f'Epoch [{epoch + 1}/{num_epoch}], Val Accuracy: {val_accuracy:.4f}')
               
        result = {
            "train_loss": avg_epoch_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "train_f1": train_f1,
            "val_f1": val_f1
        }
        
    return result
        
def validate(model, val_dataloader, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    criterion = CrossEntropyLoss()
    model.eval()
    running_loss = 0
    all_val_labels = []
    all_val_predictions = []
    with torch.no_grad():
        for embeddings, labels in tqdm(val_dataloader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_val_labels.extend(labels.tolist())
            all_val_predictions.extend(torch.argmax(outputs, 1).tolist())
    avg_val_loss = running_loss/len(val_dataloader)
    print("Classification report:\n", classification_report(all_val_labels, all_val_predictions, zero_division = True))
    return avg_val_loss, all_val_labels, all_val_predictions
    
def load_cifar10(is_train = True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    return dataset

def extract_cnn_features(dataset, device = "cuda" if torch.cuda.is_available() else "cpu"):
    resnet50 = torchvision.models.resnet34(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1]) 
    resnet50 = resnet50.to(device)
    resnet50.eval()  # Set to evaluation mode
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    list_features = []
    list_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = resnet50(images)
            outputs = outputs.view(outputs.size(0), -1)
            list_features.append(outputs.cpu())
            list_labels.append(labels.cpu())

    list_features = torch.cat(list_features, axis = 0).numpy()
    list_labels = torch.cat(list_labels, axis = 0).numpy()
    
    
    
    df_dataset = pd.DataFrame(list_features)
    df_dataset["labels"] = list_labels
    return df_dataset

def split_client_data_indices(targets, client1_classes, client2_classes, client3_classes):
    client1_indices = []
    client2_indices = []
    client3_indices = []
    # Split classes existed in client 1
    for cls in client1_classes:
        class_indices = np.where(np.array(targets) == cls)[0]
        if cls == 0:
            split = np.array_split(class_indices, 2)
            client1_indices.append(split[0])
            client3_indices.append(split[1])
        else:
            split = np.array_split(class_indices, 3)
            client1_indices.append(split[0])
            client2_indices.append(split[1])
            client3_indices.append(split[2])

    # Split classes existed in client 2 but not in client 1
    for cls in client2_classes:
        if cls not in client1_classes:
            class_indices = np.where(np.array(targets) == cls)[0]
            split = np.array_split(class_indices, 2)
            client2_indices.append(split[0])
            client3_indices.append(split[1])

    # The remain is in client 3
    for cls in client3_classes:
        if cls not in client1_classes and cls not in client2_classes:
            class_indices = np.where(np.array(targets) == cls)[0]
            client3_indices.append(class_indices)

    # Concatenate client's data indices
    client1_indices = np.concatenate(client1_indices)
    client2_indices = np.concatenate(client2_indices)
    client3_indices = np.concatenate(client3_indices)

    return [client1_indices, client2_indices, client3_indices]

def split_dataset_for_clients(dataset):
    client1_classes = list(range(0, 5))
    client2_classes = list(range(1, 10))
    client3_classes = list(range(10))

    client_indices = split_client_data_indices(dataset.targets, client1_classes, client2_classes, client3_classes)

    client1_dataset = Subset(dataset, client_indices[0])
    client2_dataset = Subset(dataset, client_indices[1])
    client3_dataset = Subset(dataset, client_indices[2])

    return client1_dataset, client2_dataset, client3_dataset

def load_dataset_from_csv(dataset_path):
    dataset = pd.read_csv(dataset_path)
    data = torch.tensor(dataset.iloc[:, :-1].values, dtype=torch.float32)  # Các cột feature
    labels = torch.tensor(dataset['labels'].values, dtype=torch.long)  # Cột nhãn
    dataset = CustomDataset(data, labels)

    return dataset

def save_dataset_to_csv(dataset, filename):
    dataset.to_csv(filename, index=False)
    
def train_val_split(dataset, val_size=0.2):
    train_size = int((1 - val_size) * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], torch.Generator().manual_seed(42))
    return train_dataset, val_dataset

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    

