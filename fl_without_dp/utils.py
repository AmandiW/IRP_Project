import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


# Custom Dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['diagnosis']

        if self.transform:
            image = self.transform(image)

        return image, label


# Data loading function
def load_data(img_dir, labels_path, num_clients=3, batch_size=32):
    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Split data for clients (IID distribution)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        client_df = remaining_df.sample(n=client_size, random_state=42 + i)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)  # Last client gets remaining data

    # Create train/test splits for each client
    client_data = []
    for client_df in client_dfs:
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42)

        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        client_data.append((train_loader, test_loader))

    return client_data


# Model creation function
def create_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    return model