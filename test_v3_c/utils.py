import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


class RetinopathyDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['id_code'] + '.png')
        image = Image.open(img_name)
        label = torch.tensor(self.data.iloc[idx]['diagnosis'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_data_loaders(image_dir, labels_file, num_clients=3, batch_size=32, test_size=0.2):
    """Create IID data distribution across clients"""
    transform = get_transforms()
    full_dataset = RetinopathyDataset(image_dir, labels_file, transform)

    # Split into train and test
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        stratify=[full_dataset[i][1] for i in range(len(full_dataset))],
        random_state=42
    )

    # Create test loader
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split training data for clients (IID distribution)
    client_size = len(train_idx) // num_clients
    client_loaders = []

    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else len(train_idx)
        client_idx = train_idx[start_idx:end_idx]

        client_dataset = torch.utils.data.Subset(full_dataset, client_idx)
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)

    return client_loaders, test_loader


def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model