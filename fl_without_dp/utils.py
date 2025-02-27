import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
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

        # Check if images exist and keep only valid entries
        valid_indices = []
        for idx in range(len(labels_df)):
            img_id = labels_df.iloc[idx]['id_code']
            img_path = os.path.join(img_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                valid_indices.append(idx)

        self.labels_df = labels_df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        # Handle potential file errors
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels_df.iloc[idx]['diagnosis']

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            # This prevents crashes but logs the error
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, self.labels_df.iloc[idx]['diagnosis']


# Data loading function
def load_data(img_dir, labels_path, num_clients=3, batch_size=32):
    # Data transformation with additional augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Check class distribution
    class_counts = labels_df['diagnosis'].value_counts()
    print(f"Class distribution in dataset: {class_counts}")

    num_classes = 2
    print(f"Using binary classification (num_classes={num_classes})")

    # Determine if stratification is needed
    if min(class_counts) >= 10:  # Ensure enough samples for stratification
        stratify = labels_df['diagnosis']
    else:
        stratify = None
        print("Warning: Some classes have very few samples, not using stratification")

    # Split data for clients (IID distribution)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        # Use stratify=remaining_df['diagnosis'] to maintain class distribution
        client_df = remaining_df.sample(n=client_size, random_state=42 + i)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)  # Last client gets remaining data

    # Create train/test splits for each client
    client_data = []
    for client_df in client_dfs:
        # Use stratified split to maintain class distribution
        train_df, test_df = train_test_split(
            client_df,
            test_size=0.2,
            random_state=42,
            stratify=client_df['diagnosis'] if stratify is not None else None
        )

        train_dataset = RetinopathyDataset(img_dir, train_df, train_transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        client_data.append((train_loader, test_loader))

        # Print client dataset stats
        print(f"Client dataset - Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        print(f"Train class distribution: {train_df['diagnosis'].value_counts()}")
        print(f"Test class distribution: {test_df['diagnosis'].value_counts()}")

    return client_data, num_classes


# Model creation function
def create_model(num_classes=2):
    # Use a smaller weight decay to prevent overfitting
    model = models.resnet18(pretrained=True)

    # Freeze early layers to prevent overfitting
    for param in list(model.parameters())[:-4]:  # Freeze all but the last few layers
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Add dropout to reduce overfitting
        nn.Linear(num_features, num_classes)
    )
    return model