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

class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

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
        

def load_dataset(data_path):
    dataset = pd.read_csv(data_path)
    dataset = CustomDataset(dataset)
    return dataset


def train(model, train_dataloader, val_dataloader, criterion, optimizer,
          num_epoch = 10, show_step = 1, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    for epoch in range(num_epoch):
        model.train()
        print("Training epoch ", epoch + 1)
        
        running_loss = 0
        
        for embeddings, labels in tqdm(train_dataloader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_epoch_train_loss = running_loss/len(train_dataloader)
        # print(f'Epoch [{epoch + 1}/{num_epoch}], Training Loss: {avg_epoch_train_loss:.4f}')
        avg_val_loss, all_val_labels, all_val_predictions = validate(model, val_dataloader, criterion)
        # print(f'Epoch [{epoch + 1}/{num_epoch}], Validation Loss: {avg_val_loss:.4f}')        
        if (epoch%show_step == 0 or epoch == num_epoch - 1):
            print(classification_report(all_val_labels, all_val_predictions, zero_division = True))

def validate(model, val_dataloader, criterion = CrossEntropyLoss(), device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
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

def split_dataset_for_clients(dataset) -> pd.DataFrame:
    client1_classes = list(range(0, 5))
    client2_classes = list(range(1, 10))
    client3_classes = list(range(10))

    client_indices = split_client_data_indices(dataset.targets, client1_classes, client2_classes, client3_classes)

    client1_dataset = Subset(dataset, client_indices[0])
    client2_dataset = Subset(dataset, client_indices[1])
    client3_dataset = Subset(dataset, client_indices[2])

    return client1_dataset, client2_dataset, client3_dataset

def read_dataset_from_csv(dataset_path):
    dataset = pd.read_csv(dataset_path)
    data = torch.tensor(dataset.iloc[:, :-1].values, dtype=torch.float32)  # Các cột feature
    labels = torch.tensor(dataset['labels'].values, dtype=torch.long)  # Cột nhãn
    dataset = CustomDataset(data, labels)

    return dataset

def train_val_split(dataset, val_size=0.2):
    train_size = int((1 - val_size) * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], torch.Generator().manual_seed(42))
    return train_dataset, val_dataset
    
