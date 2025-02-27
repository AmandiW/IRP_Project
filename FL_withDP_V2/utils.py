import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        score = -val_loss  # Higher score is better

        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop


# Data loading function
def load_data(img_dir, labels_path, num_clients=3, batch_size=32):
    # Get logger
    logger = logging.getLogger("DataLoader")

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Log dataset statistics
    class_distribution = labels_df['diagnosis'].value_counts()
    logger.info(f"Total dataset size: {len(labels_df)}")
    logger.info(f"Class distribution: {class_distribution.to_dict()}")

    # Split data for clients (IID distribution)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        # Use different random seeds for each client to avoid data leakage
        client_df = remaining_df.sample(n=client_size, random_state=42 + i * 100)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)  # Last client gets remaining data

    # Create train/test splits for each client
    client_data = []
    for i, client_df in enumerate(client_dfs):
        # Use different random seeds for train/test splits
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42 + i * 10)

        # Log client data statistics
        train_class_dist = train_df['diagnosis'].value_counts().to_dict()
        test_class_dist = test_df['diagnosis'].value_counts().to_dict()

        logger.info(f"Client {i} - Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        logger.info(f"Client {i} - Train class distribution: {train_class_dist}")
        logger.info(f"Client {i} - Test class distribution: {test_class_dist}")

        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        client_data.append((train_loader, test_loader, len(train_df), len(test_df)))

    return client_data


# Model creation function
def create_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    return model


# Create privacy-compatible PyTorch module
def create_dp_compatible_model():
    from opacus.validators import ModuleValidator

    model = create_model()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model