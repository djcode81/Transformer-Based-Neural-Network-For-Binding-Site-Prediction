# Transformer-Based Protein Binding Site Prediction

A lightweight transformer-based neural network for rapid protein-ligand binding site prediction through knowledge distillation from P2Rank.

## Overview

This project implements knowledge distillation to transfer binding site prediction capabilities from P2Rank (a slower but accurate method) to an efficient transformer-based neural network. The resulting model achieves comparable accuracy to P2Rank with a 5× speedup, making it suitable for high-throughput protein structure analysis.

## Key Features

- **5× faster** than P2Rank (0.201 seconds vs ~1 second per protein)
- **82.65% validation accuracy** with balanced performance across classes
- Processes AlphaFold predicted structures (CIF format)
- Transformer architecture with multi-head attention
- Handles class imbalance through focal loss
- Protein-based cross-validation for robust evaluation

## Architecture

The model utilizes a transformer-based architecture that captures long-range dependencies between residues through multi-head attention mechanisms. It processes multiple feature types:
- Amino acid one-hot encodings
- Physicochemical properties
- pLDDT confidence scores from AlphaFold
- Secondary structure predictions
- Sequence context windows

## Dataset

- 185 AlphaFold protein structures
- 90,098 total residues
- 16,694 binding sites (18.53%)
- Labeled using P2Rank predictions

## Requirements
python>=3.8
tensorflow>=2.8.0
biopython>=1.79
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
