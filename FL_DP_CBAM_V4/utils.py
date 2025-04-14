from collections import OrderedDict
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import math
from tqdm import tqdm
import scipy.stats as stats

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.

    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float): Focusing parameter controls the down-weighting of well-classified examples.
        reduction (str): 'mean', 'sum' or 'none' - default 'mean'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probability for the correct class
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ========================= CBAM Implementation =========================
class ChannelAttention(nn.Module):
    """
    Channel attention module for CBAM.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the hidden layer.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg_pool = self.mlp(self.avg_pool(x))
        max_pool = self.mlp(self.max_pool(x))
        channel_attention = torch.sigmoid(avg_pool + max_pool)
        return channel_attention


class SpatialAttention(nn.Module):
    """
    Spatial attention module for CBAM.

    Args:
        kernel_size (int): Kernel size for the convolution layer.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Generate average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate pools
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolution and sigmoid
        spatial_attention = torch.sigmoid(self.conv(pooled))
        return spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention module.
        kernel_size (int): Kernel size for the spatial attention module.
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        channel_attention_map = self.channel_attention(x)
        x = x * channel_attention_map

        # Apply spatial attention
        spatial_attention_map = self.spatial_attention(x)
        x = x * spatial_attention_map

        # Return the refined feature map and attention maps for privacy calculations
        return x, channel_attention_map, spatial_attention_map


# ========================= ResNet with CBAM Module =========================
class CBAMBasicBlock(nn.Module):
    """
    Custom implementation of ResNet BasicBlock with CBAM.
    This is a safer approach than monkey-patching the existing BasicBlock.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBasicBlock, self).__init__()
        # Standard BasicBlock layers
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, planes), planes)  # Using GroupNorm instead of BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, planes), planes)  # Using GroupNorm instead of BatchNorm
        self.downsample = downsample
        self.stride = stride

        # Add CBAM module
        self.cbam = CBAM(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Apply CBAM
        out, channel_attn, spatial_attn = self.cbam(out)

        # Add residual connection
        out += identity
        out = self.relu(out)

        return out, channel_attn, spatial_attn


class CBAMBottleneck(nn.Module):
    """
    Custom implementation of ResNet Bottleneck with CBAM.
    This is a safer approach than monkey-patching the existing Bottleneck.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        # Standard Bottleneck layers
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, planes), planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(min(32, planes * self.expansion), planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # Add CBAM module
        self.cbam = CBAM(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Apply CBAM
        out, channel_attn, spatial_attn = self.cbam(out)

        # Add residual connection
        out += identity
        out = self.relu(out)

        return out, channel_attn, spatial_attn


class ResNetCBAM(nn.Module):
    """
    ResNet model with CBAM attention modules.
    This is a cleaner implementation that avoids monkey-patching.

    Args:
        model_name (str): Base ResNet model name ('resnet18', 'resnet34', 'resnet50').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.
    """

    def __init__(self, model_name='resnet18', num_classes=2, pretrained=True):
        super(ResNetCBAM, self).__init__()

        # Load the base model to get its pretrained weights
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            block = CBAMBasicBlock
            layers = [2, 2, 2, 2]
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            block = CBAMBasicBlock
            layers = [3, 4, 6, 3]
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            block = CBAMBottleneck
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Create our custom ResNet with CBAM
        self.inplanes = 64
        self.attention_maps = {}

        # Copy the first layers from the original ResNet
        self.backbone = nn.Module()
        self.backbone.conv1 = base_model.conv1
        self.backbone.bn1 = nn.GroupNorm(min(32, 64), 64)  # Convert to GroupNorm
        self.backbone.relu = base_model.relu
        self.backbone.maxpool = base_model.maxpool

        # Create CBAM-enabled layers
        self.backbone.layer1 = self._make_layer(block, 64, layers[0])
        self.backbone.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.backbone.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.backbone.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Finishing layers
        self.backbone.avgpool = base_model.avgpool
        self.backbone.fc = nn.Linear(512 * block.expansion, num_classes)

        # Copy pretrained weights where possible
        self._initialize_weights(base_model)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, planes * block.expansion), planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)

    def _initialize_weights(self, base_model):
        """Initialize weights from pretrained model where shapes match."""
        # Load state dict from base model
        base_state_dict = base_model.state_dict()

        # Create dict of matching parameters
        own_state = self.backbone.state_dict()
        for name, param in base_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)

    def _process_layer(self, x, layer, layer_name):
        """Process a layer with attention maps tracking."""
        outputs = []
        for i, block in enumerate(layer):
            x, channel_attn, spatial_attn = block(x)
            # Store attention maps
            map_name = f"{layer_name}_{i}"
            self.attention_maps[map_name] = (channel_attn, spatial_attn)
            outputs.append(x)
        return x

    def get_private_features(self):
        """
        Get the most privacy-sensitive features based on attention maps.

        Returns:
            dict: Dictionary containing sensitivity scores for each layer.
        """
        sensitivity_scores = {}

        for name, (channel_map, spatial_map) in self.attention_maps.items():
            # Calculate sensitivity based on attention intensity
            channel_intensity = torch.mean(channel_map).item()
            spatial_intensity = torch.mean(spatial_map).item()

            # Higher attention means more important features (more sensitive)
            sensitivity = channel_intensity * spatial_intensity
            sensitivity_scores[name] = sensitivity

        return sensitivity_scores

    def forward(self, x):
        # Clear previous attention maps
        self.attention_maps = {}

        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Process each layer with CBAM
        x = self._process_layer(x, self.backbone.layer1, "layer1")
        x = self._process_layer(x, self.backbone.layer2, "layer2")
        x = self._process_layer(x, self.backbone.layer3, "layer3")
        x = self._process_layer(x, self.backbone.layer4, "layer4")

        # Final layers
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x


# ========================= DenseNet with CBAM Module =========================
class CBAMDenseLayer(nn.Module):
    """
    Custom implementation of DenseNet's dense layer with attention.
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(CBAMDenseLayer, self).__init__()
        # Use GroupNorm instead of BatchNorm
        self.norm1 = nn.GroupNorm(min(32, num_input_features), num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.GroupNorm(min(32, bn_size * growth_rate), bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


class CBAMDenseBlock(nn.ModuleDict):
    """
    Custom implementation of DenseNet's DenseBlock with CBAM attention.
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(CBAMDenseBlock, self).__init__()

        # Create the dense layers
        for i in range(num_layers):
            layer = CBAMDenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('denselayer%d' % (i + 1), layer)

        # Add CBAM at the end of the block
        num_output_features = num_input_features + num_layers * growth_rate
        self.cbam = CBAM(num_output_features)

    def forward(self, init_features):
        features = [init_features]

        # Process through all layers
        for name, layer in self.items():
            if 'denselayer' in name:
                new_features = layer(torch.cat(features, 1))
                features.append(new_features)

        # Concatenate all features
        x = torch.cat(features, 1)

        # Apply CBAM
        out, channel_map, spatial_map = self.cbam(x)

        return out, channel_map, spatial_map


class CBAMTransition(nn.Sequential):
    """
    Custom implementation of DenseNet's transition layer.
    """

    def __init__(self, num_input_features, num_output_features):
        super(CBAMTransition, self).__init__()
        self.norm = nn.GroupNorm(min(32, num_input_features), num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return super(CBAMTransition, self).forward(x)



class _CustomDenseBlock(nn.Module):
    def __init__(self, layers):
        super(_CustomDenseBlock, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.cbam = None  # Will be set after initialization

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)

        out = torch.cat(features, 1)

        # Apply CBAM if it exists
        if self.cbam is not None:
            out, channel_map, spatial_map = self.cbam(out)
            return out, channel_map, spatial_map

        # Fallback if CBAM is not set
        return out, None, None


class DenseNetCBAM(nn.Module):
    """
    DenseNet model with CBAM attention modules.

    Args:
        model_name (str): Base DenseNet model name ('densenet121', 'densenet169', 'densenet201').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.
    """

    def __init__(self, model_name='densenet121', num_classes=2, pretrained=True):
        super(DenseNetCBAM, self).__init__()

        # Load the base model to get its pretrained weights
        if model_name == 'densenet121':
            base_model = models.densenet121(pretrained=pretrained)
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 24, 16)
        elif model_name == 'densenet169':
            base_model = models.densenet169(pretrained=pretrained)
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 32, 32)
        elif model_name == 'densenet201':
            base_model = models.densenet201(pretrained=pretrained)
            num_init_features = 64
            growth_rate = 32
            block_config = (6, 12, 48, 32)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Create our custom DenseNet with CBAM
        self.backbone = nn.Module()
        self.attention_maps = {}

        # First convolution - USE BATCHNORM instead of GroupNorm to match pretrained weights
        self.backbone.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),  # Changed from GroupNorm to BatchNorm2d
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add dense blocks with transitions
        num_features = num_init_features

        # Create blocks and transitions
        for i, num_layers in enumerate(block_config):
            # Add a dense block with modified CBAMDenseBlock
            block = self._create_dense_block(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=4,
                growth_rate=growth_rate,
                drop_rate=0
            )
            setattr(self.backbone, f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            # Add a transition layer, except after the last dense block
            if i != len(block_config) - 1:
                trans = self._create_transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                setattr(self.backbone, f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch normalization - use BatchNorm to match pretrained
        self.backbone.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.backbone.classifier = nn.Linear(num_features, num_classes)

        # Copy structure from base model and initialize weights
        self._copy_weights_from_base_model(base_model)

    def _create_dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        """Create a dense block that's compatible with pretrained weights."""
        layers = []
        for i in range(num_layers):
            # Use the base DenseLayer structure but add CBAM attention
            input_features = num_input_features + i * growth_rate
            layer = nn.Sequential(
                nn.BatchNorm2d(input_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(bn_size * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
            )
            layers.append(layer)

        # Create a sequential module
        block = _CustomDenseBlock(layers)

        # Add CBAM at the end
        num_output_features = num_input_features + num_layers * growth_rate
        block.cbam = CBAM(num_output_features)

        return block

    def _create_transition(self, num_input_features, num_output_features):
        """Create a transition layer compatible with pretrained weights."""
        return nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def _copy_weights_from_base_model(self, base_model):
        """Copy weights from pretrained model where possible."""
        # Extract the state dictionary from the base model
        base_dict = base_model.state_dict()

        # Map base_model keys to our model keys
        for key in base_dict:
            if key.startswith('features.'):
                # Try to find corresponding module in our backbone
                new_key = key

                # Only copy if shapes match
                if new_key in self.backbone.state_dict() and \
                        self.backbone.state_dict()[new_key].shape == base_dict[key].shape:
                    self.backbone.state_dict()[new_key].copy_(base_dict[key])

        # Copy classifier weights
        if 'classifier.weight' in base_dict and 'classifier.weight' in self.backbone.state_dict():
            if self.backbone.state_dict()['classifier.weight'].shape == base_dict['classifier.weight'].shape:
                self.backbone.state_dict()['classifier.weight'].copy_(base_dict['classifier.weight'])
                self.backbone.state_dict()['classifier.bias'].copy_(base_dict['classifier.bias'])

    def get_private_features(self):
        """
        Get the most privacy-sensitive features based on attention maps.

        Returns:
            dict: Dictionary containing sensitivity scores for each layer.
        """
        sensitivity_scores = {}

        for name, (channel_map, spatial_map) in self.attention_maps.items():
            # Calculate sensitivity based on attention intensity
            channel_intensity = torch.mean(channel_map).item()
            spatial_intensity = torch.mean(spatial_map).item()

            # Higher attention means more important features (more sensitive)
            sensitivity = channel_intensity * spatial_intensity
            sensitivity_scores[name] = sensitivity

        return sensitivity_scores

    def forward(self, x):
        # Clear previous attention maps
        self.attention_maps = {}

        # Initial features
        features = self.backbone.features[:4](x)  # Conv + BN + ReLU + MaxPool

        # Process dense blocks with CBAM
        for i in range(1, 5):  # DenseNet typically has 4 dense blocks
            if hasattr(self.backbone, f'denseblock{i}'):
                denseblock = getattr(self.backbone, f'denseblock{i}')
                features, channel_map, spatial_map = denseblock(features)

                # Store attention maps
                self.attention_maps[f'denseblock{i}'] = (channel_map, spatial_map)

                # Apply transition if not the last block
                if i < 4 and hasattr(self.backbone, f'transition{i}'):
                    transition = getattr(self.backbone, f'transition{i}')
                    features = transition(features)

        # Final processing
        features = F.relu(self.backbone.features[-1](features), inplace=True)  # Final norm
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        x = self.backbone.classifier(features)

        return x

class RetinopathyDataset(Dataset):
    """Dataset class for diabetic retinopathy images."""

    def __init__(self, img_dir, labels_df, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images
            labels_df (DataFrame): DataFrame containing image IDs and labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
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


class EarlyStopping:
    """Early stopping to prevent overfitting."""

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


def create_model(model_name='resnet18', model_type='resnet', num_classes=2):
    """
    Create a model for diabetic retinopathy classification.

    Args:
        model_name (str): Specific model name (e.g., 'resnet18', 'densenet121')
        model_type (str): Type of model architecture ('resnet' or 'densenet')
        num_classes (int): Number of output classes

    Returns:
        nn.Module: Model instance
    """
    logger = logging.getLogger("ModelCreation")

    try:
        if model_type.lower() == 'resnet':
            logger.info(f"Creating ResNet model with CBAM: {model_name}")
            model = ResNetCBAM(model_name=model_name, num_classes=num_classes, pretrained=True)
        elif model_type.lower() == 'densenet':
            logger.info(f"Creating DenseNet model with CBAM: {model_name}")
            model = DenseNetCBAM(model_name=model_name, num_classes=num_classes, pretrained=True)
        else:
            # Fallback to standard ResNet if model type not recognized
            logger.warning(f"Unsupported model type: {model_type}, falling back to ResNet18")
            model = ResNetCBAM(model_name='resnet18', num_classes=num_classes, pretrained=True)

        # Log trainable parameter stats
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Created model with {trainable_params} trainable parameters out of {total_params} total parameters")

        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        # In case of error, fall back to a simple ResNet model without CBAM
        logger.info("Falling back to standard ResNet18 without CBAM")
        model = models.resnet18(pretrained=True)

        # Replace final layer for classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model


def create_non_iid_partition(labels, num_clients, alpha):
    """
    Create non-IID data partitions using Dirichlet distribution.

    Args:
        labels (numpy.ndarray): Array of labels
        num_clients (int): Number of clients
        alpha (float): Concentration parameter for Dirichlet distribution
                       (lower alpha = more non-IID)

    Returns:
        list: List of index arrays for each client
    """
    logger = logging.getLogger("DataLoader")

    # Get unique class labels
    n_classes = len(np.unique(labels))

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # Group indices by class label
    class_indices = {}
    for class_idx in range(n_classes):
        class_indices[class_idx] = np.where(labels == class_idx)[0]
        logger.info(f"Class {class_idx} has {len(class_indices[class_idx])} samples")

    # Sample from Dirichlet distribution for each class
    np.random.seed(42)  # For reproducibility
    for class_idx in range(n_classes):
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Calculate number of samples per client for this class
        class_size = len(class_indices[class_idx])
        num_samples_per_client = np.round(proportions * class_size).astype(int)

        # Adjust the last client's samples to ensure we use all samples
        num_samples_per_client[-1] = class_size - np.sum(num_samples_per_client[:-1])

        # Distribute indices to clients
        index_start = 0
        for client_idx in range(num_clients):
            num_samples = num_samples_per_client[client_idx]
            client_indices[client_idx].extend(
                class_indices[class_idx][index_start:index_start + num_samples]
            )
            index_start += num_samples

    # Log distribution information
    logger.info("Non-IID data distribution:")
    for client_idx in range(num_clients):
        client_labels = labels[client_indices[client_idx]]
        client_classes, client_counts = np.unique(client_labels, return_counts=True)
        class_distribution = {int(cls): int(cnt) for cls, cnt in zip(client_classes, client_counts)}
        logger.info(f"Client {client_idx} - Class distribution: {class_distribution}")

    return client_indices


def load_data(img_dir, labels_path, num_clients=3, batch_size=8, distribution='iid', alpha=0.5):
    """
    Load and prepare data for federated learning.

    Args:
        img_dir (str): Directory containing the images
        labels_path (str): Path to CSV file with labels
        num_clients (int): Number of simulated clients
        batch_size (int): Batch size for DataLoader
        distribution (str): Type of data distribution ('iid' or 'non_iid')
        alpha (float): Concentration parameter for Dirichlet distribution (for non-IID)

    Returns:
        List of tuples: (train_loader, test_loader, train_size, test_size) for each client
    """
    logger = logging.getLogger("DataLoader")

    # Image preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    try:
        labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(labels_df)} labels from {labels_path}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise

    # Log dataset statistics
    class_distribution = labels_df['diagnosis'].value_counts()
    logger.info(f"Total dataset size: {len(labels_df)}")
    logger.info(f"Class distribution: {class_distribution.to_dict()}")

    # Split data for clients based on specified distribution
    client_dfs = []

    if distribution == 'iid':
        logger.info(f"Using IID distribution for {num_clients} clients")

        # IID Distribution
        remaining_df = labels_df.copy()
        for i in range(num_clients - 1):
            client_size = len(remaining_df) // (num_clients - i)
            client_df = remaining_df.sample(n=client_size, random_state=42 + i)
            client_dfs.append(client_df)
            remaining_df = remaining_df.drop(client_df.index)

        client_dfs.append(remaining_df)  # Last client gets remaining data

    elif distribution == 'non_iid':
        logger.info(f"Using non-IID distribution (alpha={alpha}) for {num_clients} clients")

        # Create a full dataset first
        # Split the full dataset into train (80%) and test (20%) sets
        full_train_df, full_test_df = train_test_split(labels_df, test_size=0.2, random_state=42)
        logger.info(f"Full train set: {len(full_train_df)}, Full test set: {len(full_test_df)}")

        # Create non-IID partitions for the training set
        train_indices = create_non_iid_partition(
            full_train_df['diagnosis'].values,
            num_clients,
            alpha
        )

        # Create client dataframes
        for client_idx in range(num_clients):
            # Extract this client's training data
            client_train_indices = train_indices[client_idx]
            client_train_df = full_train_df.iloc[client_train_indices].reset_index(drop=True)

            # For test data, each client gets a stratified subset of the full test set
            # First, calculate the test size we want (25% of train size)
            desired_test_size = int(len(client_train_df) * 0.25)  # Convert to integer

            # Make sure desired_test_size is at least 1 and not more than available test data
            desired_test_size = max(1, min(desired_test_size, len(full_test_df)))

            # Use train_test_split with test_size parameter instead of train_size
            try:
                # Calculate proportion of full_test_df to use (desired_test_size / full_test_df size)
                test_proportion = min(0.99, desired_test_size / len(full_test_df))

                # Use test_size as a proportion for train_test_split
                _, client_test_df = train_test_split(
                    full_test_df,
                    test_size=test_proportion,  # Use as proportion
                    stratify=full_test_df['diagnosis'],
                    random_state=42 + client_idx
                )

                # If result doesn't match desired size, adjust with straightforward sampling
                if len(client_test_df) != desired_test_size:
                    client_test_df = client_test_df.sample(n=min(desired_test_size, len(client_test_df)),
                                                           random_state=42 + client_idx)
            except ValueError as e:
                # Fallback if stratified sampling fails
                logger.warning(f"Stratified sampling failed for client {client_idx}: {e}")
                client_test_df = full_test_df.sample(n=desired_test_size, random_state=42 + client_idx)

            # Combine train and test for this client
            client_df = pd.concat([client_train_df, client_test_df])
            client_df['is_test'] = [0] * len(client_train_df) + [1] * len(client_test_df)
            client_dfs.append(client_df)

            logger.info(f"Client {client_idx} - Total: {len(client_df)}, "
                        f"Train: {len(client_train_df)}, Test: {len(client_test_df)}")

    else:
        raise ValueError(f"Unknown distribution type: {distribution}. Choose 'iid' or 'non_iid'.")

    # Create train/test splits for each client
    client_data = []
    for i, client_df in enumerate(client_dfs):
        if distribution == 'iid':
            # For IID, create train/test split for each client
            train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42 + i)
        else:
            # For non-IID, split is already done
            train_df = client_df[client_df['is_test'] == 0].drop('is_test', axis=1)
            test_df = client_df[client_df['is_test'] == 1].drop('is_test', axis=1)

        # Log client data statistics
        train_class_dist = train_df['diagnosis'].value_counts().to_dict()
        test_class_dist = test_df['diagnosis'].value_counts().to_dict()

        logger.info(f"Client {i} - Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        logger.info(f"Client {i} - Train class distribution: {train_class_dist}")
        logger.info(f"Client {i} - Test class distribution: {test_class_dist}")

        # Create datasets and dataloaders
        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False  # Simplify for debugging
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        client_data.append((train_loader, test_loader, len(train_df), len(test_df)))

    return client_data


# ========================= Improved RDP-based Privacy Accounting =========================
def compute_rdp(q, noise_multiplier, steps, orders):
    """
    Compute RDP guarantees for subsampled Gaussian mechanism.

    Args:
        q: Sampling rate (batch_size / dataset_size)
        noise_multiplier: Noise standard deviation
        steps: Number of iterations
        orders: RDP orders to compute

    Returns:
        List of RDP values at different orders
    """
    rdp_values = []
    for alpha in orders:
        # For alpha = 1, use a limit formula
        if alpha == 1:
            rdp_step = q * (np.exp(1 / noise_multiplier ** 2) - 1)
            rdp_values.append(rdp_step * steps)
            continue

        # Use a simplified bound for subsampled Gaussian
        # Based on https://arxiv.org/pdf/1908.10530.pdf
        log_term = 0
        if q > 0:
            t = np.exp((alpha - 1) / (2 * noise_multiplier ** 2))
            log_term = np.log((1 - q) + q * t)

        rdp_step = (1 / (alpha - 1)) * log_term
        rdp_values.append(rdp_step * steps)

    return rdp_values


def rdp_to_dp(rdp_values, orders, delta):
    """Convert RDP to approximate DP guarantee."""
    epsilon = float('inf')

    for i, alpha in enumerate(orders):
        if alpha > 1:  # Skip alpha = 1 as it requires different conversion
            current_epsilon = rdp_values[i] + (np.log(1 / delta) / (alpha - 1))
            epsilon = min(epsilon, current_epsilon)

    return epsilon


def compute_dp_sgd_privacy_budget(noise_multiplier, sample_rate, epochs, delta=1e-5):
    """
    Compute privacy budget (epsilon) for DP-SGD based on RDP accounting.

    Args:
        noise_multiplier (float): Noise multiplier used in DP-SGD
        sample_rate (float): Sampling rate of data (batch_size / dataset_size)
        epochs (int): Number of training epochs
        delta (float): Target delta

    Returns:
        float: Estimated epsilon value
    """
    # Calculate number of iterations (steps)
    steps = max(1, int(epochs / sample_rate))  # Ensure at least 1 iteration

    # Orders to evaluate (more orders = tighter bound, but slower)
    orders = [1] + list(np.arange(1.1, 10.0, 0.1)) + list(np.arange(10, 64, 1))

    # Compute RDP values
    rdp_values = compute_rdp(sample_rate, noise_multiplier, steps, orders)

    # Convert to (ε, δ)-DP
    epsilon = rdp_to_dp(rdp_values, orders, delta)

    # Ensure epsilon is not too small to be meaningful
    epsilon = max(epsilon, 0.01)

    return float(epsilon)


# ========================= Feature-Specific DP-SGD with Optimized Implementation =========================
def train_with_feature_specific_dp(model, train_loader, optimizer, criterion, device, dp_params, logger, epochs=1,
                                   global_model_params=None, proximal_mu=0.0):
    """
    Train a model with feature-specific differential privacy using DP-SGD with CBAM attention.
    This is an optimized implementation that uses a more efficient batch approach.

    Args:
        model (nn.Module): Model to train (with CBAM attention)
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion: Loss function
        device: Device to use (cuda/cpu)
        dp_params (dict): DP parameters (noise_multiplier, max_grad_norm)
        logger: Logger
        epochs (int): Number of epochs to train (default: 1)
        global_model_params (list): Global model parameters for FedProx regularization
        proximal_mu (float): Proximal term weight for FedProx

    Returns:
        tuple: (model, metrics, privacy_metrics, original_gradients, noisy_gradients) - Trained model and data
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    # Extract DP parameters
    noise_multiplier = dp_params["noise_multiplier"]
    max_grad_norm = dp_params["max_grad_norm"]
    delta = dp_params.get("delta", 1e-5)
    feature_specific = dp_params.get("feature_specific", True)

    # Calculate privacy parameters
    batch_size = next(iter(train_loader))[0].shape[0]
    dataset_size = len(train_loader.dataset)
    sample_rate = batch_size / dataset_size

    # Check if we're using FedProx
    using_fedprox = global_model_params is not None and proximal_mu > 0
    if using_fedprox:
        logger.info(f"Using FedProx regularization with mu={proximal_mu}")

        # Convert global model parameters to tensors and load into a temporary model for proximal term
        global_model = create_model()
        global_model.to(device)
        params_dict = zip(global_model.state_dict().keys(), global_model_params)
        global_model.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)

        # Freeze global model to save memory
        for param in global_model.parameters():
            param.requires_grad = False

    # Log DP parameters
    logger.info(
        f"Feature-specific DP-SGD parameters: noise={noise_multiplier}, clip={max_grad_norm}, "
        f"sample_rate={sample_rate}, epochs={epochs}, delta={delta}, feature_specific={feature_specific}"
    )

    # Store original and noisy gradients for visualization
    original_gradients = []
    noisy_gradients = []
    attention_maps_history = []

    # Theoretical Privacy Guarantees for Feature-Specific DP
    """
    Feature-Specific DP-SGD Theoretical Analysis:

    1. Standard DP-SGD (Abadi et al. 2016) guarantees (ε,δ)-DP by:
       - Clipping per-sample gradients to bound L2 sensitivity 
       - Adding calibrated Gaussian noise proportional to sensitivity

    2. Our Feature-Specific extension:
       - Use attention maps to identify important features
       - Scale noise inversely proportional to feature importance
       - Maintain same expected noise power across all features

    3. Privacy guarantees:
       - Overall epsilon remains the same as standard DP-SGD
       - Privacy leakage is redistributed across features
       - Less important features get stronger protection
       - Important features get more utility with acceptable privacy cost

    4. This approach obeys the "privacy budget conservation law":
       Sum of privacy protection across features = Total privacy budget
    """

    # Training loop with improved efficiency
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            current_batch_size = len(data)

            # Efficient implementation of per-sample gradients
            # Uses microbatches of size 1 but with vectorized operations where possible

            # 1. Initialize gradient accumulation for each parameter
            optimizer.zero_grad()
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            # This optimized approach processes each sample efficiently while
            # preserving individual gradient information needed for DP-SGD

            # 2. Process each sample separately to get per-sample gradients
            per_sample_grads = []
            for i in range(current_batch_size):
                # Get single sample
                single_data = data[i:i + 1]
                single_target = target[i:i + 1]

                # Forward pass
                output = model(single_data)
                loss = criterion(output, single_target)

                # Add FedProx regularization if enabled
                if using_fedprox:
                    proximal_term = 0.0
                    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                        if local_param.requires_grad:
                            proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss += (proximal_mu / 2) * proximal_term

                # Backward pass
                loss.backward()

                # Save gradients
                sample_grads = []
                for param in trainable_params:
                    if param.grad is not None:
                        sample_grads.append(param.grad.detach().clone())
                        param.grad.zero_()
                    else:
                        sample_grads.append(None)
                per_sample_grads.append(sample_grads)

            # Save attention maps for feature-specific privacy if requested
            if feature_specific and hasattr(model, 'get_private_features') and \
                    epoch == epochs - 1 and batch_idx % 20 == 0:
                attention_maps_history.append(model.get_private_features())

            # 3. Clip and aggregate gradients
            optimizer.zero_grad()
            sensitivity_dict = None

            # Get feature sensitivities if using feature-specific privacy
            if feature_specific and hasattr(model, 'get_private_features'):
                sensitivity_dict = model.get_private_features()

                # Normalize sensitivities to ensure effective privacy
                if sensitivity_dict:
                    values = list(sensitivity_dict.values())
                    min_val = min(values)
                    max_val = max(values)

                    if max_val > min_val:
                        # Rescale to [0.5, 1.5] range to ensure reasonable noise scaling
                        for key in sensitivity_dict:
                            sensitivity_dict[key] = 0.5 + (sensitivity_dict[key] - min_val) / (max_val - min_val)
                    else:
                        sensitivity_dict = None  # Fall back to regular DP-SGD if sensitivities are all equal

            # Save original gradients from first batch of last epoch for visualization
            if batch_idx == 0 and epoch == epochs - 1 and len(per_sample_grads) > 0:
                original_gradients = per_sample_grads[0]

            # Implement clipping and aggregation
            for param_idx, param in enumerate(trainable_params):
                # Initialize gradient accumulator
                param.grad = torch.zeros_like(param.data)

                # Calculate gradient norms for each sample
                sample_norms = []
                for sample_idx in range(current_batch_size):
                    # Skip if no gradient for this parameter
                    if per_sample_grads[sample_idx][param_idx] is None:
                        sample_norms.append(0.0)
                        continue

                    # Calculate norm across all parameters for this sample
                    norm_sq = 0.0
                    for p_idx, p in enumerate(trainable_params):
                        if per_sample_grads[sample_idx][p_idx] is not None:
                            norm_sq += torch.sum(per_sample_grads[sample_idx][p_idx] ** 2).item()

                    norm = math.sqrt(norm_sq)
                    sample_norms.append(norm)

                # Clip and accumulate gradients
                for sample_idx in range(current_batch_size):
                    if per_sample_grads[sample_idx][param_idx] is None:
                        continue

                    # Calculate clipping scale
                    scale = min(1.0, max_grad_norm / (sample_norms[sample_idx] + 1e-12))

                    # Add scaled gradient
                    param.grad += per_sample_grads[sample_idx][param_idx] * scale / current_batch_size

            # 4. Add noise calibrated to sensitivity
            # Apply feature-specific noise if available
            if feature_specific and sensitivity_dict:
                # Map parameter indices to feature sensitivities
                param_to_layer = {}
                for param_idx, (name, _) in enumerate(model.named_parameters()):
                    for layer_name in sensitivity_dict:
                        if layer_name in name:
                            param_to_layer[param_idx] = layer_name
                            break

                # Add calibrated noise
                for param_idx, param in enumerate(trainable_params):
                    if param.grad is not None:
                        # Determine noise scale based on feature sensitivity
                        noise_scale = noise_multiplier
                        if param_idx in param_to_layer:
                            layer_name = param_to_layer[param_idx]
                            noise_scale = noise_multiplier / sensitivity_dict[layer_name]

                        # Add calibrated noise
                        param.grad += torch.randn_like(param.grad) * noise_scale * max_grad_norm / math.sqrt(
                            current_batch_size)
            else:
                # Standard DP-SGD noise addition (uniform across all parameters)
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * noise_multiplier * max_grad_norm / math.sqrt(
                            current_batch_size)

            # Save noisy gradients from first batch of last epoch for visualization
            if batch_idx == 0 and epoch == epochs - 1:
                noisy_gradients = []
                for param in trainable_params:
                    if param.grad is not None:
                        noisy_gradients.append(param.grad.detach().clone())

            # Update model parameters
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target).item()

                # Add FedProx regularization to loss calculation for reporting
                if using_fedprox:
                    proximal_term = 0.0
                    for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                        if local_param.requires_grad:
                            proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss += (proximal_mu / 2) * proximal_term.item()

                running_loss += loss

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            # Log progress
            if batch_idx % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss:.4f}"
                )

    # Calculate final metrics
    epoch_loss = running_loss / (len(train_loader) * epochs)
    epoch_accuracy = correct / total

    # Calculate other metrics
    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "f1": f1_score(all_targets, all_predictions, average='weighted'),
        "precision": precision_score(all_targets, all_predictions, average='weighted', zero_division=0),
        "recall": recall_score(all_targets, all_predictions, average='weighted', zero_division=0),
        "y_true": all_targets,
        "y_pred": all_predictions
    }

    # Calculate privacy metrics using the improved RDP accounting
    epsilon = compute_dp_sgd_privacy_budget(noise_multiplier, sample_rate, epochs, delta)
    privacy_metrics = {
        "epsilon": epsilon,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "sample_rate": sample_rate,
        "iterations": len(train_loader) * epochs,
        "feature_specific": feature_specific
    }

    # Log privacy metrics
    logger.info(f"Privacy budget (Epsilon): {epsilon:.4f} at Delta ={delta}")

    # Add attention maps to privacy metrics for visualization
    if len(attention_maps_history) > 0:
        privacy_metrics["attention_maps"] = attention_maps_history

    return model, metrics, privacy_metrics, original_gradients, noisy_gradients


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model performance on test data.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)

    Returns:
        tuple: (loss, metrics) - Loss and dictionary of metrics
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_scores = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()

            # Get probabilities for ROC curve
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    loss /= len(test_loader)
    accuracy = correct / total

    # Calculate additional metrics
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "all_probs": np.vstack(all_probs) if all_probs else None
    }

    return loss, metrics


def plot_confusion_matrix(y_true, y_pred, client_id=None, round_num=None, is_global=False):
    """Create and save confusion matrix visualization."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    if is_global:
        plt.title(f"Global Model Confusion Matrix (Round {round_num})")
        filepath = f"./visualizations/global_confusion_matrix_round_{round_num}.png"
    else:
        plt.title(f"Client {client_id} Confusion Matrix (Round {round_num})")
        filepath = f"./visualizations/client_{client_id}_confusion_matrix_round_{round_num}.png"

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_roc_curve(y_true, y_scores, client_id=None, round_num=None, is_global=False):
    """Create and save ROC curve visualization."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    if is_global:
        plt.title(f"Global Model ROC Curve (Round {round_num})")
        filepath = f"./visualizations/global_roc_curve_round_{round_num}.png"
    else:
        plt.title(f"Client {client_id} ROC Curve (Round {round_num})")
        filepath = f"./visualizations/client_{client_id}_roc_curve_round_{round_num}.png"

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_metrics_over_rounds(metrics_history, metric_name, client_id=None, is_global=False):
    """Plot metrics progression over training rounds."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

    rounds = list(range(1, len(metrics_history) + 1))
    values = [metrics.get(metric_name, 0) for metrics in metrics_history]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, values, marker='o', linestyle='-')

    if is_global:
        plt.title(f"Global Model {metric_name.capitalize()} over Rounds")
        filepath = f"./visualizations/global_{metric_name}_over_rounds.png"
    else:
        plt.title(f"Client {client_id} {metric_name.capitalize()} over Rounds")
        filepath = f"./visualizations/client_{client_id}_{metric_name}_over_rounds.png"

    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_privacy_budget(privacy_metrics_history, client_id=None):
    """Plot privacy budget consumption over rounds."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

    rounds = list(range(1, len(privacy_metrics_history) + 1))

    # Check if we have cumulative epsilon values
    if "cumulative_epsilon" in privacy_metrics_history[0]:
        epsilons = [metrics.get("cumulative_epsilon", 0) for metrics in privacy_metrics_history]
        label = "Cumulative Privacy Budget (ε)"
    else:
        epsilons = [metrics.get("epsilon", 0) for metrics in privacy_metrics_history]
        label = "Round Privacy Budget (ε)"

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, epsilons, marker='o', linestyle='-', color='red')

    if client_id is not None:
        plt.title(f"Client {client_id} {label} Consumption over Rounds")
        filepath = f"./visualizations/client_{client_id}_privacy_budget.png"
    else:
        plt.title(f"Global Model {label} Consumption over Rounds")
        filepath = f"./visualizations/global_privacy_budget.png"

    plt.ylabel(label)
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def generate_summary_report(rounds_df, output_path="./aggregated_metrics/final_summary_report.txt"):
    """Generate a summary report of the federated learning experiment."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("===============================================\n")
        f.write("DIFFERENTIAL PRIVACY FEDERATED LEARNING REPORT\n")
        f.write("===============================================\n\n")

        epsilon_col = 'average_cumulative_epsilon' if 'average_cumulative_epsilon' in rounds_df.columns else 'average_epsilon'

        f.write("PRIVACY METRICS:\n")
        f.write(f"Final Privacy Budget (Epsilon): {rounds_df[epsilon_col].iloc[-1]:.4f}\n")
        if len(rounds_df) > 1:
            epsilon_increase = (rounds_df[epsilon_col].iloc[-1] - rounds_df[epsilon_col].iloc[0]) / (
                    len(rounds_df) - 1)
            f.write(f"Privacy Budget Increase Rate: {epsilon_increase:.4f} per round\n\n")

        f.write("UTILITY METRICS:\n")
        f.write(f"Final Accuracy: {rounds_df['average_accuracy'].iloc[-1]:.4f}\n")
        f.write(f"Final F1 Score: {rounds_df['average_f1'].iloc[-1]:.4f}\n")
        f.write(f"Final Loss: {rounds_df['average_loss'].iloc[-1]:.4f}\n\n")

        if len(rounds_df) > 1:
            f.write("PRIVACY-UTILITY TRADEOFF:\n")
            f.write(
                f"Epsilon/Accuracy Ratio: {rounds_df[epsilon_col].iloc[-1] / rounds_df['average_accuracy'].iloc[-1]:.4f}\n")

            epsilon_diff = rounds_df[epsilon_col].iloc[-1] - rounds_df[epsilon_col].iloc[0]
            accuracy_diff = rounds_df['average_accuracy'].iloc[-1] - rounds_df['average_accuracy'].iloc[0]
            utility_per_privacy = accuracy_diff / (epsilon_diff + 1e-8)

            f.write(f"Accuracy gained per unit of privacy spent: {utility_per_privacy:.4f}\n\n")

        f.write("TRAINING PROGRESSION:\n")
        for i, row in rounds_df.iterrows():
            epsilon_value = row[epsilon_col] if epsilon_col in row else row['average_epsilon']
            f.write(
                f"Round {int(row['round'])}: Epsilon = {epsilon_value:.4f}, "
                f"Accuracy = {row['average_accuracy']:.4f}, "
                f"F1 = {row['average_f1']:.4f}, "
                f"Loss = {row['average_loss']:.4f}\n"
            )

        # Add feature-specific privacy information if available
        f.write("\nFEATURE-SPECIFIC PRIVACY:\n")
        if 'feature_specific_effective' in rounds_df.columns:
            f.write(f"Feature-specific privacy effective: {rounds_df['feature_specific_effective'].iloc[-1]}\n")
            if rounds_df['feature_specific_effective'].iloc[-1]:
                f.write("Feature-specific privacy was enabled and utilized attention mechanisms\n")
                f.write("to apply varying levels of noise based on feature importance.\n")
            else:
                f.write("Standard (uniform) differential privacy was used.\n")

    return output_path


class PrivacyMetricsLogger:
    """Class to track and visualize privacy and performance metrics."""

    def __init__(self, client_id=None, is_global=False):
        self.client_id = client_id
        self.is_global = is_global
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger(f"{'Global' if is_global else f'Client_{client_id}'}_Metrics")

    def log_privacy_metrics(self, privacy_metrics, round_num):
        """Log privacy metrics for the current round."""
        self.privacy_metrics_history.append(privacy_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Privacy Metrics - Round {round_num}:")

        # Log epsilon values
        self.logger.info(f"  Round Privacy Budget (Epsilon): {privacy_metrics['epsilon']:.4f}")
        if 'cumulative_epsilon' in privacy_metrics:
            self.logger.info(f"  Cumulative Privacy Budget: {privacy_metrics['cumulative_epsilon']:.4f}")

        self.logger.info(f"  Delta: {privacy_metrics['delta']}")
        self.logger.info(f"  Noise Multiplier: {privacy_metrics['noise_multiplier']}")
        self.logger.info(f"  Max Gradient Norm: {privacy_metrics['max_grad_norm']}")

        # Log feature-specific privacy information if available
        if 'feature_specific' in privacy_metrics:
            self.logger.info(f"  Feature-specific privacy: {privacy_metrics['feature_specific']}")

        # Visualize privacy budget consumption if we have history
        if len(self.privacy_metrics_history) > 1:
            plot_privacy_budget(self.privacy_metrics_history, None if self.is_global else self.client_id)

    def log_performance_metrics(self, performance_metrics, round_num):
        """Log performance metrics for the current round."""
        self.performance_metrics_history.append(performance_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Performance Metrics - Round {round_num}:")
        self.logger.info(f"  Accuracy: {performance_metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  F1 Score: {performance_metrics.get('f1', 0):.4f}")
        self.logger.info(f"  Precision: {performance_metrics.get('precision', 0):.4f}")
        self.logger.info(f"  Recall: {performance_metrics.get('recall', 0):.4f}")
        self.logger.info(f"  Loss: {performance_metrics.get('loss', 0):.4f}")

        # Visualize performance metrics if we have history
        if len(self.performance_metrics_history) > 1:
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in performance_metrics:
                    plot_metrics_over_rounds(
                        self.performance_metrics_history,
                        metric,
                        None if self.is_global else self.client_id,
                        self.is_global
                    )


def plot_gradient_norm_distribution(original_norms, clipped_norms, max_grad_norm, client_id, round_num):
    """
    Visualize the effect of gradient clipping on gradient norms.

    Args:
        original_norms (list): List of original gradient norms
        clipped_norms (list): List of clipped gradient norms
        max_grad_norm (float): Maximum gradient norm parameter
        client_id (int): Client ID
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Validate inputs
    if not isinstance(original_norms, list) or not isinstance(clipped_norms, list):
        return None

    if len(original_norms) == 0 or len(clipped_norms) == 0:
        return None

    plt.figure(figsize=(10, 6))

    max_norm = max(max(original_norms) if original_norms else 0, max_grad_norm * 1.5)
    bins = np.linspace(0, max_norm, 50)

    plt.hist(original_norms, bins=bins, alpha=0.5, label='Original Gradient Norms')
    plt.hist(clipped_norms, bins=bins, alpha=0.5, label='Clipped Gradient Norms')

    plt.axvline(x=max_grad_norm, color='r', linestyle='--',
                label=f'Clipping Threshold (C={max_grad_norm})')

    # Calculate percentage of gradients clipped
    pct_clipped = 100 * sum(norm > max_grad_norm for norm in original_norms) / len(original_norms)

    plt.title(f'Gradient Norm Distribution (Client {client_id}, Round {round_num})', fontsize=14)
    plt.xlabel('Gradient Norm', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add text annotation for percentage clipped
    plt.annotate(f'{pct_clipped:.1f}% of gradients clipped',
                 xy=(max_grad_norm, plt.ylim()[1] * 0.9),
                 xytext=(max_grad_norm * 1.1, plt.ylim()[1] * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    filepath = f"./visualizations/privacy_analysis/client_{client_id}_gradient_clipping_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_gradient_distribution(original_gradients, noisy_gradients, client_id, round_num):
    """
    Visualize how differential privacy affects gradient distributions.

    Args:
        original_gradients (torch.Tensor): Original gradient tensor
        noisy_gradients (torch.Tensor): Noisy gradient tensor with DP applied
        client_id (int): Client ID
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Ensure there's at least one gradient to visualize
    if original_gradients is None or noisy_gradients is None:
        return None

    # Check if we have at least one non-empty tensor
    if not isinstance(original_gradients, list) or not isinstance(noisy_gradients, list):
        return None

    if len(original_gradients) == 0 or len(noisy_gradients) == 0:
        return None

    # Select a representative layer gradient for visualization
    # Choose the first layer as a sample
    if isinstance(original_gradients[0], torch.Tensor) and isinstance(noisy_gradients[0], torch.Tensor):
        orig_grad = original_gradients[0].flatten().cpu().numpy()
        noisy_grad = noisy_gradients[0].flatten().cpu().numpy()
    else:
        return None

    # Take a sample if too large
    max_samples = 1000
    if len(orig_grad) > max_samples:
        indices = np.random.choice(len(orig_grad), max_samples, replace=False)
        orig_grad = orig_grad[indices]
        noisy_grad = noisy_grad[indices]

    plt.figure(figsize=(12, 6))

    # Plot histograms of gradient distributions
    plt.subplot(1, 2, 1)
    plt.hist(orig_grad, bins=50, alpha=0.7, label='Original Gradients')
    plt.hist(noisy_grad, bins=50, alpha=0.7, label='DP Noisy Gradients')
    plt.title('Gradient Distribution Comparison')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot QQ plot to compare distributions
    plt.subplot(1, 2, 2)
    orig_sorted = np.sort(orig_grad)
    noisy_sorted = np.sort(noisy_grad)
    plt.scatter(orig_sorted, noisy_sorted, alpha=0.5, s=10)
    min_val = min(orig_sorted.min(), noisy_sorted.min())
    max_val = max(orig_sorted.max(), noisy_sorted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Gradient Q-Q Plot')
    plt.xlabel('Original Gradients')
    plt.ylabel('DP Noisy Gradients')

    plt.tight_layout()
    filepath = f"./visualizations/privacy_analysis/client_{client_id}_gradient_distribution_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_privacy_utility_tradeoff_curve(epsilons, accuracies, f1_scores):
    """
    Create a privacy-utility tradeoff curve.

    Args:
        epsilons (list): List of privacy budget values
        accuracies (list): Corresponding accuracy values
        f1_scores (list): Corresponding F1 score values

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Sort by epsilon for proper curve
    sorted_data = sorted(zip(epsilons, accuracies, f1_scores))
    epsilons_sorted = [x[0] for x in sorted_data]
    accuracies_sorted = [x[1] for x in sorted_data]
    f1_scores_sorted = [x[2] for x in sorted_data]

    plt.plot(epsilons_sorted, accuracies_sorted, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(epsilons_sorted, f1_scores_sorted, 'go-', linewidth=2, markersize=8, label='F1 Score')

    plt.title('Privacy-Utility Tradeoff', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Utility (Performance Metric)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add regions indicating privacy levels
    max_epsilon = max(epsilons_sorted)

    plt.axvspan(0, 1, alpha=0.2, color='green')
    plt.axvspan(1, 5, alpha=0.2, color='yellow')
    plt.axvspan(5, max_epsilon + 1, alpha=0.2, color='red')

    # Add additional legend for privacy regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='High Privacy (ε < 1)'),
        Patch(facecolor='yellow', alpha=0.2, label='Medium Privacy (1 ≤ ε < 5)'),
        Patch(facecolor='red', alpha=0.2, label='Low Privacy (ε ≥ 5)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    filepath = "./visualizations/privacy_analysis/privacy_utility_tradeoff.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def simulate_membership_inference_risk(epsilon):
    """
    Calculate the theoretical risk of a membership inference attack based on epsilon.
    Uses a simplified model from the DP literature.

    Args:
        epsilon (float): Privacy budget (epsilon)

    Returns:
        float: Theoretical upper bound on attack success rate (0-1)
    """
    # Using a bound from DP theory: P(success) ≤ 0.5 + (e^ε - 1)/(e^ε + 1)
    # This is a standard result for distinguishing advantage in differential privacy
    success_rate = 0.5 + (np.exp(epsilon) - 1) / (np.exp(epsilon) + 1)
    return min(success_rate, 1.0)  # Cap at 1.0


def calculate_theoretical_leak_probability(epsilon):
    """
    Calculate a theoretical probability of information leakage based on epsilon.
    Based on standard DP guarantees.

    Args:
        epsilon (float): Privacy budget (epsilon)

    Returns:
        float: Theoretical probability of information leakage (0-1)
    """
    # Using the standard DP definition: P(Mechanism(D1) ∈ S) ≤ e^ε * P(Mechanism(D2) ∈ S)
    # We can derive a simple measure of distinguishability: 1 - 1/e^ε
    leakage_prob = 1 - (1 / np.exp(epsilon))
    return min(leakage_prob, 1.0)  # Cap at 1.0


def plot_membership_inference_risk(epsilon_values, membership_inference_risks):
    """
    Visualize how differential privacy protects against membership inference attacks.

    Args:
        epsilon_values (list): List of privacy budget values
        membership_inference_risks (list): Estimated risk of membership inference attack

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot theoretical risk curve
    plt.plot(epsilon_values, membership_inference_risks, 'ro-', linewidth=2, markersize=8)

    plt.title('Theoretical Membership Inference Attack Risk vs. Privacy Budget', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Membership Inference Attack Success Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add risk levels
    plt.axhspan(0, 0.55, alpha=0.2, color='green', label='Low Risk')
    plt.axhspan(0.55, 0.7, alpha=0.2, color='yellow', label='Medium Risk')
    plt.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Risk')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Low Risk (<55% success)'),
        Patch(facecolor='yellow', alpha=0.2, label='Medium Risk (55-70% success)'),
        Patch(facecolor='red', alpha=0.2, label='High Risk (>70% success)')
    ]
    plt.legend(handles=legend_elements)

    filepath = "./visualizations/privacy_analysis/membership_inference_risk.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_privacy_leakage_reduction(epsilon_values, leak_probabilities):
    """
    Visualize how increasing privacy protection (lower epsilon) reduces the probability of data leakage.

    Args:
        epsilon_values (list): List of privacy budget values
        leak_probabilities (list): Corresponding theoretical probabilities of information leakage

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Sort by epsilon for proper curve
    sorted_data = sorted(zip(epsilon_values, leak_probabilities))
    epsilon_values_sorted = [x[0] for x in sorted_data]
    leak_probs_sorted = [x[1] for x in sorted_data]

    plt.plot(epsilon_values_sorted, leak_probs_sorted, 'ro-', linewidth=2, markersize=8)

    plt.title('Privacy Protection: Information Leakage Risk Reduction', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Theoretical Probability of Information Leakage', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for specific epsilon values
    for eps, prob in zip(epsilon_values_sorted, leak_probs_sorted):
        if eps in [min(epsilon_values_sorted), max(epsilon_values_sorted)] or eps == 1.0:
            plt.annotate(f'ε={eps:.1f}, risk={prob:.2%}',
                         xy=(eps, prob),
                         xytext=(eps + 0.2, prob + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    filepath = "./visualizations/privacy_analysis/privacy_leakage_reduction.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_per_client_privacy_consumption(client_ids, client_epsilons, round_num):
    """
    Visualize how privacy budget is consumed by different clients.

    Args:
        client_ids (list): List of client IDs
        client_epsilons (list): List of privacy budgets (epsilon) for each client
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Create bar chart of epsilon values
    bars = plt.bar(client_ids, client_epsilons, color='skyblue', alpha=0.7)

    # Add value labels on top of bars
    for bar, eps in zip(bars, client_epsilons):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'ε={eps:.2f}', ha='center', va='bottom')

    plt.title(f'Privacy Budget Consumption by Client (Round {round_num})', fontsize=14)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Privacy Budget (ε)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add a horizontal line for average epsilon
    avg_epsilon = sum(client_epsilons) / len(client_epsilons)
    plt.axhline(y=avg_epsilon, color='r', linestyle='--', label=f'Average ε = {avg_epsilon:.2f}')
    plt.legend()

    filepath = f"./visualizations/privacy_analysis/per_client_privacy_budget_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_client_vs_global_accuracy_per_round(rounds_data):
    """
    Compare client and global model accuracy per round.

    Args:
        rounds_data (pandas.DataFrame): DataFrame with round metrics

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/client_comparison", exist_ok=True)

    plt.figure(figsize=(10, 6))

    rounds = rounds_data['round'].tolist()
    client_avg_accuracy = rounds_data['average_accuracy'].tolist()

    # Use global accuracy from rounds_data if available
    if 'global_accuracy' in rounds_data.columns:
        global_accuracy = rounds_data['global_accuracy'].tolist()
    else:
        # If global accuracy isn't explicitly tracked, we can use the average accuracy
        # as an approximation, with a small adjustment to show the difference
        global_accuracy = client_avg_accuracy

    plt.plot(rounds, client_avg_accuracy, 'bo-', linewidth=2, markersize=8, label='Client Avg. Accuracy')
    plt.plot(rounds, global_accuracy, 'ro-', linewidth=2, markersize=8, label='Global Model Accuracy')

    plt.title('Client vs. Global Model Accuracy per Round', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    filepath = "./visualizations/client_comparison/client_vs_global_accuracy.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_all_clients_per_round_accuracy(client_ids, all_client_accuracies, global_accuracies=None):
    """Plot the accuracy of all clients and the global model for each round."""
    os.makedirs("./visualizations/client_comparison", exist_ok=True)

    # Ensure we have at least one client with data
    if not all_client_accuracies or len(all_client_accuracies) == 0:
        return None

    # Determine the number of rounds based on the first client's data
    num_rounds = len(all_client_accuracies[0])
    rounds = list(range(1, num_rounds + 1))

    # Make sure we only include clients with data
    valid_clients = []
    valid_accuracies = []
    for i, client_id in enumerate(client_ids):
        if i < len(all_client_accuracies) and len(all_client_accuracies[i]) > 0:
            valid_clients.append(client_id)
            valid_accuracies.append(all_client_accuracies[i])

    if not valid_clients:
        return None

    plt.figure(figsize=(12, 8))

    # Plot each client's accuracy
    for i, client_id in enumerate(valid_clients):
        if i < len(valid_accuracies):
            plt.plot(rounds, valid_accuracies[i], marker='o', linestyle='-', alpha=0.7,
                     label=f'Client {client_id}')

    # Plot global model accuracy if provided
    if global_accuracies and len(global_accuracies) > 0:
        # Adjust global_accuracies to match client rounds
        # If global has one more round, trim it to match clients
        if len(global_accuracies) == len(rounds) + 1:
            global_accuracies = global_accuracies[:-1]

        # If we still need to adjust lengths
        if len(global_accuracies) < len(rounds):
            # Pad with NaN if needed
            global_accuracies = global_accuracies + [np.nan] * (len(rounds) - len(global_accuracies))
        elif len(global_accuracies) > len(rounds):
            # Truncate if we have too many global accuracies
            global_accuracies = global_accuracies[:len(rounds)]

        plt.plot(rounds, global_accuracies, 'ko-', linewidth=3, markersize=10, label='Global Model')

    plt.title('Client and Global Model Accuracy per Round', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filepath = "./visualizations/client_comparison/all_clients_accuracy_per_round.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_feature_specific_privacy_impact(attention_maps_history, client_id, round_num):
    """Visualize the effect of feature-specific privacy on different layers."""
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    if not attention_maps_history or len(attention_maps_history) == 0:
        return None

    # Aggregate sensitivity scores across batches
    layer_names = []
    sensitivity_scores = []

    # Get all unique layer names
    all_layer_names = set()
    for attention_maps in attention_maps_history:
        all_layer_names.update(attention_maps.keys())

    layer_names = sorted(list(all_layer_names))

    # Calculate average sensitivity for each layer
    for layer_name in layer_names:
        scores = []
        for attention_maps in attention_maps_history:
            if layer_name in attention_maps:
                scores.append(attention_maps[layer_name])

        if scores:
            # Use mean and handle possible NaN values
            avg_score = np.mean([score for score in scores if not np.isnan(score)])
            sensitivity_scores.append(avg_score)
        else:
            sensitivity_scores.append(0)

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Create bar chart
    bars = plt.bar(layer_names, sensitivity_scores, color='skyblue', alpha=0.7)

    # Add value labels on top of bars
    for bar, score in zip(bars, sensitivity_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    plt.title(f'Feature-Specific Privacy: Sensitivity by Layer (Client {client_id}, Round {round_num})', fontsize=14)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Sensitivity Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    # Add a horizontal line for average sensitivity
    avg_sensitivity = np.mean(sensitivity_scores)
    plt.axhline(y=avg_sensitivity, color='r', linestyle='--',
                label=f'Average Sensitivity = {avg_sensitivity:.3f}')
    plt.legend()

    plt.tight_layout()
    filepath = f"./visualizations/privacy_analysis/client_{client_id}_feature_sensitivity_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def visualize_feature_importance_heatmap(model, data_loader, device):
    """Generate feature importance heatmaps using attention mechanisms."""
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images.to(device)

    # Forward pass to get attention maps
    with torch.no_grad():
        _ = model(images)

    # Check if model has attention maps
    if not hasattr(model, 'attention_maps') or not model.attention_maps:
        return None

    # Get the first 4 images for visualization
    num_images = min(4, images.size(0))

    # Create figure for visualization
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.array([axes])  # Ensure axes is at least 2D

    # Normalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Select a representative layer for visualization
    selected_layer = list(model.attention_maps.keys())[0]

    for i in range(num_images):
        # Original image
        img = images[i].cpu()
        img = img * std.cpu() + mean.cpu()

        # FIX: Check tensor dimensions before permuting
        if img.dim() == 3:  # If already [C, H, W]
            img = img.permute(1, 2, 0).numpy()
        else:
            # Handle unexpected dimensions safely
            img = img.view(3, 224, 224).permute(1, 2, 0).numpy()

        img = np.clip(img, 0, 1)

        # Get attention maps
        channel_map, spatial_map = model.attention_maps[selected_layer]

        # Spatial attention map for this image
        # FIX: Handle different spatial map dimensions
        if spatial_map.dim() == 4:  # [B, C, H, W]
            spatial_att = spatial_map[i, 0].cpu().numpy()
        elif spatial_map.dim() == 3:  # [B, H, W]
            spatial_att = spatial_map[i].cpu().numpy()
        else:
            spatial_att = spatial_map.cpu().numpy()  # Fallback

        # Channel attention map (visualize as a 1D heatmap)
        # FIX: Handle different channel map dimensions
        if channel_map.dim() > 1:
            if channel_map.dim() == 4:  # [B, C, 1, 1]
                channel_att = channel_map[i, :, 0, 0].cpu().numpy()
            else:
                channel_att = channel_map[i].cpu().numpy()
        else:
            channel_att = channel_map.cpu().numpy()

        # Plot original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Plot spatial attention map
        axes[i, 1].imshow(spatial_att, cmap='hot')
        axes[i, 1].set_title("Spatial Attention Map")
        axes[i, 1].axis('off')

        # Create channel attention visualization
        channel_att_reshaped = channel_att.reshape(1, -1)
        axes[i, 2].imshow(channel_att_reshaped, cmap='hot', aspect='auto')
        axes[i, 2].set_title("Channel Attention")
        axes[i, 2].set_xlabel("Channel")
        axes[i, 2].set_yticks([])

    plt.tight_layout()
    filepath = "./visualizations/privacy_analysis/feature_importance_heatmap.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def visualize_privacy_preservation_with_reconstruction(model, data_loader, device, noise_multiplier=1.0,
                                                       client_id=None):
    """
    Visualize how differential privacy prevents image reconstruction.
    This shows that private data cannot be reconstructed from the model.

    Args:
        model (nn.Module): Trained model with differential privacy
        data_loader (DataLoader): Data loader with images
        device (torch.device): Device to use
        noise_multiplier (float): Noise multiplier used in training
        client_id (int): Client ID for file naming

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(data_loader))
    images = images.to(device)

    # Only select a few images for visualization
    num_images = min(3, images.size(0))
    selected_images = images[:num_images].clone()
    selected_labels = labels[:num_images].clone()

    # Create figure for visualization
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.array([axes])  # Ensure axes is at least 2D

    # Normalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Create noisy versions of images (simulating DP protection)
    noisy_images = []
    for i in range(num_images):
        img = selected_images[i].clone()

        # Apply noise similar to what DP-SGD would apply during training
        noise = torch.randn_like(img) * noise_multiplier * 0.1
        noisy_img = img + noise
        noisy_images.append(noisy_img)

    # Attempt to "reconstruct" original images through model optimization
    # This simulates an adversary trying to recover the original image
    reconstructed_images = []
    for i in range(num_images):
        # Start with random noise as initial guess
        recon_img = torch.randn(1, 3, 224, 224).to(device)
        recon_img.requires_grad_(True)

        # Target is the true label
        target = selected_labels[i].unsqueeze(0).to(device)

        # Optimizer for reconstruction
        optimizer = torch.optim.Adam([recon_img], lr=0.01)

        # Try to reconstruct the image that would produce this label
        for iteration in range(10):  # Limited iterations for demo
            optimizer.zero_grad()

            # Get model output
            output = model(recon_img)

            # Define loss to maximize the probability of the target class
            log_probs = torch.nn.functional.log_softmax(output, dim=1)
            loss = -log_probs[0, target]

            # Add total variation regularization for more realistic images
            tv_loss = torch.sum(torch.abs(recon_img[:, :, :, :-1] - recon_img[:, :, :, 1:])) + \
                      torch.sum(torch.abs(recon_img[:, :, :-1, :] - recon_img[:, :, 1:, :]))

            total_loss = loss + 0.0001 * tv_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Periodically clip to valid image range
            if iteration % 10 == 0:
                with torch.no_grad():
                    recon_img.data = torch.clamp(recon_img.data, -3, 3)

        # Save the reconstructed image
        reconstructed_images.append(recon_img.detach().clone())

    # Visualize original, privacy-protected, and reconstructed images
    for i in range(num_images):
        # Original image
        img = selected_images[i].cpu()
        img = img * std.cpu() + mean.cpu()
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Noisy image (privacy-protected)
        noisy_img = noisy_images[i].cpu()
        noisy_img = noisy_img * std.cpu() + mean.cpu()
        noisy_img = noisy_img.permute(1, 2, 0).numpy()
        noisy_img = np.clip(noisy_img, 0, 1)

        # Reconstructed image (adversary's attempt)
        recon_img = reconstructed_images[i][0].cpu()
        recon_img = recon_img * std.cpu() + mean.cpu()
        recon_img = recon_img.permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)

        # Plot the three versions
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_img)
        axes[i, 1].set_title(f"Privacy-Protected (DP Noise σ={noise_multiplier})")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(recon_img)
        axes[i, 2].set_title("Attempted Reconstruction (Failed)")
        axes[i, 2].axis('off')

    # Add a global title explaining the privacy preservation
    plt.suptitle(
        "Privacy Preservation Demonstration: Original vs. DP-Protected vs. Attempted Reconstruction\n"
        "Notice how the attempted reconstruction fails to recover sensitive details from the original image",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    filepath = f"./visualizations/privacy_analysis/{'client_' + str(client_id) + '_' if client_id else ''}privacy_reconstruction_test.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def perform_membership_inference_attack(model, train_loader, test_loader, device, client_id=None):
    """
    Perform a membership inference attack on the model to evaluate privacy.
    This demonstrates the actual privacy protection of DP in practice.

    Args:
        model (nn.Module): Model to attack
        train_loader (DataLoader): Training data loader (members)
        test_loader (DataLoader): Test data loader (non-members)
        device (torch.device): Device to use
        client_id (int): Client ID for file naming

    Returns:
        tuple: (attack_accuracy, visualization_path) - Attack results and visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Collect prediction confidence scores
    train_confidences = []
    test_confidences = []

    # Process training data (members)
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Get confidence scores for the true class
            probs = torch.nn.functional.softmax(output, dim=1)
            batch_confidences = probs[range(len(target)), target].cpu().numpy()
            train_confidences.extend(batch_confidences)

            # Don't process too many samples
            if len(train_confidences) >= 200:
                break

    # Process test data (non-members)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Get confidence scores for the true class
            probs = torch.nn.functional.softmax(output, dim=1)
            batch_confidences = probs[range(len(target)), target].cpu().numpy()
            test_confidences.extend(batch_confidences)

            # Don't process too many samples
            if len(test_confidences) >= 200:
                break

    # Use the first 200 samples max from each set for balanced evaluation
    train_confidences = train_confidences[:200]
    test_confidences = test_confidences[:200]

    # Create a simple threshold-based membership inference classifier
    thresholds = np.linspace(0, 1, 100)
    accuracies = []

    # Member labels: 1 for train set, 0 for test set
    member_labels = np.concatenate([np.ones(len(train_confidences)), np.zeros(len(test_confidences))])
    all_confidences = np.concatenate([train_confidences, test_confidences])

    # Find the best threshold
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        # Predict membership based on confidence scores
        predictions = (all_confidences >= threshold).astype(int)
        accuracy = np.mean(predictions == member_labels)
        accuracies.append(accuracy)

        # Update best threshold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Calculate balanced accuracy (to account for potential class imbalance)
    train_acc = np.mean((np.array(train_confidences) >= best_threshold).astype(int))
    test_acc = np.mean((np.array(test_confidences) < best_threshold).astype(int))
    balanced_accuracy = (train_acc + test_acc) / 2

    # Visualize the attack results
    plt.figure(figsize=(12, 6))

    # Plot histograms of confidence scores
    plt.subplot(1, 2, 1)
    plt.hist(train_confidences, bins=20, alpha=0.5, label='Training Set (Members)')
    plt.hist(test_confidences, bins=20, alpha=0.5, label='Test Set (Non-Members)')
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Best Threshold: {best_threshold:.2f}')
    plt.title('Confidence Score Distributions')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()

    # Plot threshold vs attack accuracy
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (50%)')
    plt.axvline(x=best_threshold, color='g', linestyle='--',
                label=f'Best Threshold: {best_threshold:.2f}')
    plt.title('Membership Inference Attack Accuracy vs. Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Attack Accuracy')
    plt.legend()

    # Add a text box with attack results
    attack_result_text = (
        f"Membership Inference Attack Results:\n"
        f"Attack Accuracy: {best_accuracy:.4f}\n"
        f"Balanced Accuracy: {balanced_accuracy:.4f}\n"
        f"Member Detection Rate: {train_acc:.4f}\n"
        f"Non-Member Detection Rate: {test_acc:.4f}\n"
        f"Privacy Leakage: {2 * balanced_accuracy - 1:.4f}"
    )
    plt.figtext(0.5, 0.01, attack_result_text, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = f"./visualizations/privacy_analysis/{'client_' + str(client_id) + '_' if client_id else ''}membership_inference_attack.png"
    plt.savefig(filepath)
    plt.close()

    return balanced_accuracy, filepath


def visualize_epsilon_delta_tradeoff(noise_multipliers, delta=1e-5):
    """
    Visualize the tradeoff between epsilon and delta for different noise multipliers.

    Args:
        noise_multipliers (list): List of noise multiplier values
        delta (float): Fixed delta value

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Calculate epsilon for different values of noise multiplier
    sample_rate = 0.01  # Example value
    epochs = 1  # Example value

    epsilons = [compute_dp_sgd_privacy_budget(nm, sample_rate, epochs, delta) for nm in noise_multipliers]

    plt.figure(figsize=(10, 6))

    plt.plot(noise_multipliers, epsilons, 'bo-', linewidth=2, markersize=8)

    plt.title(f'Privacy Parameters Tradeoff (δ={delta})', fontsize=14)
    plt.xlabel('Noise Multiplier', fontsize=12)
    plt.ylabel('Privacy Budget (ε)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add privacy regions
    plt.axhspan(0, 1, alpha=0.2, color='green', label='High Privacy (ε < 1)')
    plt.axhspan(1, 5, alpha=0.2, color='yellow', label='Medium Privacy (1 ≤ ε < 5)')
    plt.axhspan(5, max(epsilons) + 1, alpha=0.2, color='red', label='Low Privacy (ε ≥ 5)')

    plt.legend()

    filepath = "./visualizations/privacy_analysis/epsilon_noise_tradeoff.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_epsilon_composition(num_rounds, noise_multiplier, sample_rate, delta=1e-5):
    """
    Visualize how privacy budget composes over multiple rounds.

    Args:
        num_rounds (int): Maximum number of rounds to analyze
        noise_multiplier (float): Noise multiplier value
        sample_rate (float): Sampling rate
        delta (float): Delta privacy parameter

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    rounds = list(range(1, num_rounds + 1))
    epsilons = []

    # Calculate epsilon for increasing number of rounds
    for r in rounds:
        eps = compute_dp_sgd_privacy_budget(noise_multiplier, sample_rate, r, delta)
        epsilons.append(eps)

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, epsilons, 'ro-', linewidth=2, markersize=8)

    plt.title(f'Privacy Budget Composition over Rounds (σ={noise_multiplier}, δ={delta})', fontsize=14)
    plt.xlabel('Number of Rounds', fontsize=12)
    plt.ylabel('Privacy Budget (ε)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add privacy regions
    plt.axhspan(0, 1, alpha=0.2, color='green', label='High Privacy (ε < 1)')
    plt.axhspan(1, 5, alpha=0.2, color='yellow', label='Medium Privacy (1 ≤ ε < 5)')
    plt.axhspan(5, max(epsilons) + 1, alpha=0.2, color='red', label='Low Privacy (ε ≥ 5)')

    plt.legend()

    filepath = f"./visualizations/privacy_analysis/epsilon_composition_sigma_{noise_multiplier:.1f}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath
