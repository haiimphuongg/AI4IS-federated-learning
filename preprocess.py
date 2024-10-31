'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''
import pandas as pd
import torch
from torchvision import datasets, transforms, models
import numpy as np
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from skimage.feature import hog
import torch.nn as nn


def split_client_data_indices(targets, client1_classes, client2_classes, client3_classes):
    client1_indices = []
    client2_indices = []
    client3_indices = []

    # Split classes existed in client 1
    for cls in client1_classes:
        class_indices = np.where(targets == cls)[0]
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
            class_indices = np.where(targets == cls)[0]
            split = np.array_split(class_indices, 2)
            client2_indices.append(split[0])
            client3_indices.append(split[1])

    # The remain is in client 3
    for cls in client3_classes:
        if cls not in client1_classes and cls not in client2_classes:
            class_indices = np.where(targets == cls)[0]
            client3_indices.append(class_indices)

    # Concatenate client's data indices
    client1_indices = np.concatenate(client1_indices)
    client2_indices = np.concatenate(client2_indices)
    client3_indices = np.concatenate(client3_indices)

    return [client1_indices, client2_indices, client3_indices]


def split_train_val(dataset, targets, train_size=0.8, random_state=42):
    strat_split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)

    for train_idx, val_idx in strat_split.split(np.zeros(len(targets)), targets):
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

    return train_set, val_set


def extract_hog_features(image):
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    hog_feature = hog(image,
                      orientations=9,
                      pixels_per_cell=(4, 4),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      feature_vector=True,
                      channel_axis=2)
    return hog_feature


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype = torch.float32)
        self.labels = torch.tensor(labels, dtype = torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

def extract_hog_for_dataset(dataset):
    hog_features_list = [extract_hog_features(image) for image, _ in tqdm(dataset)]
    labels_list = [label for _, label in dataset]

    return CustomDataset(np.array(hog_features_list), np.array(labels_list))


def extract_cnn_features(cnn_model, dataset, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load ResNet50 pre-trained model and remove the last fully connected layer
    cnn_model = cnn_model.to(device)
    cnn_model.eval()  # Set to evaluation mode

    # DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    labels = []

    # Loop through the dataset
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = cnn_model(inputs)  # Pass through ResNet50
            outputs = outputs.view(outputs.size(0), -1)  # Flatten output to [batch_size, 2048]
            
            features.append(outputs.cpu())
            labels.append(targets.cpu())
    
    # Concatenate all the batches
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Return a new TensorDataset with extracted features and original labels
    return TensorDataset(features, labels)


def prepare_data(train_dataset, targets, extract_method = 'hog', device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_dataset_splited, val_dataset_splited = split_train_val(train_dataset, targets, train_size=0.8)
    if extract_method == 'hog':
        train_dataset_extracted = extract_hog_for_dataset(train_dataset_splited)
        val_dataset_extracted = extract_hog_for_dataset(val_dataset_splited)
    else:
        cnn_model = models.resnet34(pretrained=True)
        cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])  # Remove last fully connected layer
        cnn_model = cnn_model.to(device)
        
        train_dataset_extracted = extract_cnn_features(cnn_model, train_dataset_splited,batch_size=256)
        val_dataset_extracted = extract_cnn_features(cnn_model, val_dataset_splited, batch_size=256)
    
    return train_dataset_extracted, val_dataset_extracted


def export_csv(dataset, filename, method_extract='hog'):
    if method_extract == 'hog':
        data = dataset.data.numpy()
        labels = dataset.labels.numpy()
    else:
        data, labels = dataset.tensors
        data = data.numpy()
        labels = labels.numpy()

    df = pd.DataFrame(data)
    df['label'] = labels  # Add labels as the last column
    
    df.to_csv(filename, index=False)


def create_dataloader(dataset, batch_size=256):    
    data = dataset.iloc[:, :-1].values
    labels = dataset['label'].values
    
    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create dataloaders
    dataset = TensorDataset(data_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def save_dataset_to_csv():
    np.random.seed(42)

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    data = np.array(train_dataset.data)
    targets = np.array(train_dataset.targets)

    '''
    Create dataloader of each client
        - Client 1: airplane, automobile, bird, cat, deer, dog
        - Client 2: automobile, bird, cat, deer, dog, frog, horse, ship, truck
        - Client 3: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    ''' 

    # Define client classes
    client1_classes = list(range(0, 5))
    client2_classes = list(range(1, 10))
    client3_classes = list(range(10))

    # Split data among clients
    client_indices = split_client_data_indices(targets, client1_classes, client2_classes, client3_classes)

    client1_dataset = Subset(train_dataset, client_indices[0])
    client2_dataset = Subset(train_dataset, client_indices[1])
    client3_dataset = Subset(train_dataset, client_indices[2])

    client1_train_dataset_HOG, client1_val_dataset_HOG = prepare_data(client1_dataset, targets[client_indices[0]], 'hog')
    client2_train_dataset_HOG, client2_val_dataset_HOG = prepare_data(client2_dataset, targets[client_indices[1]], 'hog')
    client3_train_dataset_HOG, client3_val_dataset_HOG = prepare_data(client3_dataset, targets[client_indices[2]], 'hog')

    client1_train_dataset_CNN, client1_val_dataset_CNN = prepare_data(client1_dataset, targets[client_indices[0]], 'cnn')
    client2_train_dataset_CNN, client2_val_dataset_CNN = prepare_data(client2_dataset, targets[client_indices[1]], 'cnn')
    client3_train_dataset_CNN, client3_val_dataset_CNN = prepare_data(client3_dataset, targets[client_indices[2]], 'cnn')
    
    # Export file
    export_csv(client1_train_dataset_HOG, "client1_train_dataset_HOG.csv")
    export_csv(client2_train_dataset_HOG, "client2_train_dataset_HOG.csv")
    export_csv(client3_train_dataset_HOG, "client3_train_dataset_HOG.csv")
    export_csv(client1_val_dataset_HOG, "client1_val_dataset_HOG.csv")
    export_csv(client2_val_dataset_HOG, "client2_val_dataset_HOG.csv")
    export_csv(client3_val_dataset_HOG, "client3_val_dataset_HOG.csv")

    export_csv(client1_train_dataset_CNN, "client1_train_dataset_CNN.csv", 'cnn')
    export_csv(client2_train_dataset_CNN, "client2_train_dataset_CNN.csv", 'cnn')
    export_csv(client3_train_dataset_CNN, "client3_train_dataset_CNN.csv", 'cnn')
    export_csv(client1_val_dataset_CNN, "client1_val_dataset_CNN.csv", 'cnn')
    export_csv(client2_val_dataset_CNN, "client2_val_dataset_CNN.csv", 'cnn')
    export_csv(client3_val_dataset_CNN, "client3_val_dataset_CNN.csv", 'cnn')


if __name__ == "__main__":
    save_dataset_to_csv()     # Uncomment this line to export csv file
    # pass
