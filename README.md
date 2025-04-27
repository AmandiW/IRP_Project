# Enhancing Privacy in Distributed Systems for Diabetic Retinopathy Diagnosis Using Federated Learning

## Project Overview

This repository implements a novel approach to Privacy-Preserving Federated Learning for Diabetic Retinopathy diagnosis. The system incorporates Feature-Specific Differential Privacy (FS-DP) using Attention Mechanisms to enhance the privacy-utility tradeoff in medical image analysis.

## Key Features

- Feature-Specific Differential Privacy: Uses Convolutional Block Attention Module (CBAM) to apply varying levels of privacy protection based on the importance of features
- Federated Learning: Supports both FedAvg and FedProx strategies for distributed model training
- Privacy Guarantees: Implements and tracks (ε,δ)-differential privacy guarantees
- Privacy Analysis Tools: Comprehensive visualizations for privacy-utility tradeoff and membership inference risk
- Model Support: Implements privacy-enhanced versions of ResNet and DenseNet architectures

## Repository Structure

- Basic-FL-DP_Model/: Contains implementation of the baseline differential privacy federated learning model without feature-specific noise calibration.
- FS-DP_Model/: Contains the novel Feature-Specific Differential Privacy implementation that leverages attention mechanisms to apply calibrated noise based on feature importance.
- IRP_Data/: Contains the data preprocessing notebooks for the Diabetic Retinopathy dataset, including cleaning, augmentation, and exploratory data analysis.

## Feature-Specific DP Implementation Files

- FS-DP_Model/client.py: Implements the federated learning client with our novel feature-specific DP mechanism that calibrates noise based on attention-derived feature importance scores.
- FS-DP_Model/main.py: Coordinates the federated learning process with feature-specific privacy, tracking both utility metrics and privacy budget consumption.
- FS-DP_Model/server.py: Server implementation that aggregates models while preserving feature-specific privacy guarantees and calculating global privacy metrics.
- FS-DP_Model/utils.py: Core utility functions including CBAM attention modules, feature sensitivity calculation algorithms, privacy accounting, and specialized analysis tools for evaluating the privacy-utility tradeoff.
- FS-DP_Model/irp_ui.py: User-Interface that provides a comprehensive graphical environment for configuring and running Feature-Specific DP experiments.

