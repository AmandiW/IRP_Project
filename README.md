# Enhancing Privacy in Distributed Systems for Diabetic Retinopathy Diagnosis Using Federated Learning

## Project Overview

This repository implements a novel approach to Privacy-Preserving Federated Learning for Diabetic Retinopathy diagnosis. The system incorporates Feature-Specific Differential Privacy (FS-DP) using Attention Mechanisms to enhance the privacy-utility tradeoff in medical image analysis.

## Key Features

- Feature-Specific Differential Privacy: Uses Convolutional Block Attention Module (CBAM) to apply varying levels of privacy protection based on the importance of features
- Federated Learning: Supports both FedAvg and FedProx strategies for distributed model training
- Privacy Guarantees: Implements and tracks (ε,δ)-differential privacy guarantees
- Privacy Analysis Tools: Comprehensive visualizations for privacy-utility tradeoff and membership inference risk
- Model Support: Implements privacy-enhanced versions of ResNet and DenseNet architectures
