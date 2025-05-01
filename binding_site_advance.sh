#!/bin/bash
#SBATCH --job-name=binding_site_adv
#SBATCH --partition=any_cpu  # Using CPU partition to avoid cuDNN issues
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16    # Increased CPU count for faster training
#SBATCH --mem=64G
#SBATCH --time=24:00:00       # Allow up to 24 hours for training
#SBATCH --output=binding_site_adv_%j.log

# Environment setup
module purge
export PYTHONUNBUFFERED=1

# Move to project root
cd $HOME/struct/struct_final

# Activate conda environment
source ~/.bashrc
conda activate binding_site_env

echo "[$(date)] Starting advanced binding site distillation model training"

# Paths
ALPHAFOLD_DIR="alphafold_cif"
P2RANK_DIR="pockets_alphafold/cases"
OUTPUT_DIR="distillation_model_advanced"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create Python script for advanced model training
cat > train_advanced_model.py << 'EOF'
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import ResidueDepth
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, Activation
from tensorflow.keras.layers import Flatten, Concatenate, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import traceback
from sklearn.preprocessing import StandardScaler

# Force CPU usage to avoid cuDNN issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# For debugging
def log(message):
    print(message, flush=True)

# Check for CPU/GPU status
log(f"TensorFlow version: {tf.__version__}")
log(f"Using CPU only (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

# Paths
alphafold_dir = "alphafold_cif"
p2rank_dir = "pockets_alphafold/cases"
output_dir = "distillation_model_advanced"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Define amino acid dictionary for one-hot encoding
aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Amino acid grouping by physicochemical properties
aa_groups = {
    'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W'],
    'polar': ['S', 'T', 'N', 'Q', 'Y'],
    'charged_positive': ['K', 'R', 'H'],
    'charged_negative': ['D', 'E'],
    'special': ['C', 'G', 'P']
}

# Physicochemical properties
aa_properties = {
    # Hydrophobicity (Kyte-Doolittle scale)
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,

    # Normalized values for easier training
    '_hydrophobicity_min': -4.5,
    '_hydrophobicity_max': 4.5,
    
    # Volume (Å³)
    '_volume': {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.9, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    },
    '_volume_min': 60.1,
    '_volume_max': 227.8,
    
    # Charge at pH 7
    '_charge': {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    },
    
    # Polarity
    '_polarity': {
        'A': 0.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 0.0,
        'G': 0.0, 'H': 1.0, 'I': 0.0, 'K': 1.0, 'L': 0.0,
        'M': 0.0, 'N': 1.0, 'P': 0.0, 'Q': 1.0, 'R': 1.0,
        'S': 1.0, 'T': 1.0, 'V': 0.0, 'W': 0.0, 'Y': 1.0
    },
    
    # Surface exposure tendency (higher = more exposed)
    '_surface_exposure': {
        'A': 0.3, 'C': 0.2, 'D': 0.9, 'E': 0.9, 'F': 0.3,
        'G': 0.4, 'H': 0.5, 'I': 0.1, 'K': 0.9, 'L': 0.1,
        'M': 0.4, 'N': 0.8, 'P': 0.7, 'Q': 0.8, 'R': 0.9,
        'S': 0.6, 'T': 0.5, 'V': 0.2, 'W': 0.2, 'Y': 0.4
    },
    
    # Propensity for secondary structure (alpha helix)
    '_helix_propensity': {
        'A': 1.0, 'C': 0.4, 'D': 0.5, 'E': 1.0, 'F': 0.6,
        'G': 0.0, 'H': 0.6, 'I': 0.8, 'K': 0.7, 'L': 1.0,
        'M': 0.9, 'N': 0.3, 'P': 0.0, 'Q': 0.8, 'R': 0.7,
        'S': 0.3, 'T': 0.3, 'V': 0.6, 'W': 0.5, 'Y': 0.5
    }
}

# Normalize a value between min and max
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Function to convert amino acid to one-hot encoding
def one_hot_encode(amino_acid):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros(20)
    if amino_acid in amino_acids:
        idx = amino_acids.index(amino_acid)
        encoding[idx] = 1
    return encoding

# Function to get amino acid group encoding
def get_aa_group_encoding(aa):
    group_encoding = np.zeros(5)  # One for each group
    
    if aa in aa_groups['hydrophobic']:
        group_encoding[0] = 1
    elif aa in aa_groups['polar']:
        group_encoding[1] = 1
    elif aa in aa_groups['charged_positive']:
        group_encoding[2] = 1
    elif aa in aa_groups['charged_negative']:
        group_encoding[3] = 1
    elif aa in aa_groups['special']:
        group_encoding[4] = 1
        
    return group_encoding

# Function to get physicochemical properties for an amino acid
def get_physicochemical_props(aa):
    props = []
    
    # Hydrophobicity (normalized)
    if aa in aa_properties:
        hydro = normalize(
            aa_properties[aa], 
            aa_properties['_hydrophobicity_min'], 
            aa_properties['_hydrophobicity_max']
        )
        props.append(hydro)
    else:
        props.append(0.5)  # Default to middle value
        
    # Volume (normalized)
    if aa in aa_properties['_volume']:
        vol = normalize(
            aa_properties['_volume'][aa],
            aa_properties['_volume_min'],
            aa_properties['_volume_max']
        )
        props.append(vol)
    else:
        props.append(0.5)
        
    # Charge
    if aa in aa_properties['_charge']:
        # Normalize from -1,1 to 0,1
        charge = (aa_properties['_charge'][aa] + 1) / 2
        props.append(charge)
    else:
        props.append(0.5)

    # Polarity
    if aa in aa_properties['_polarity']:
        props.append(aa_properties['_polarity'][aa])
    else:
        props.append(0.5)

    # Surface exposure tendency
    if aa in aa_properties['_surface_exposure']:
        props.append(aa_properties['_surface_exposure'][aa])
    else:
        props.append(0.5)

    # Helix propensity
    if aa in aa_properties['_helix_propensity']:
        props.append(aa_properties['_helix_propensity'][aa])
    else:
        props.append(0.5)
        
    return np.array(props)

# Function to extract features from AlphaFold CIF files with neighborhood context
def extract_features_from_cif(cif_path, window_size=5):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", cif_path)
        
        # Extract basic residue features
        basic_features = {}
        residue_list = []

        # Calculate secondary structure if possible
        try:
            dssp_dict = {}
            first_model = next(structure.get_models())
            for chain in first_model:
                if len(chain) > 0:  # Only process chains with residues
                    try:
                        dssp = DSSP(first_model, cif_path, dssp='mkdssp')
                        dssp_dict.update(dssp)
                    except Exception as e:
                        log(f"Warning: Could not calculate DSSP for {cif_path}, chain {chain.id}: {e}")
        except Exception as e:
            log(f"Warning: Could not calculate DSSP for {cif_path}: {e}")
            dssp_dict = {}
            
        for model in structure:
            for chain in model:
                chain_residues = []
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        res_id = f"{chain.id}_{residue.id[1]}"
                        
                        # Get amino acid
                        res_name = residue.get_resname()
                        if res_name in aa_dict:
                            aa = aa_dict[res_name]
                            
                            # Get pLDDT score (B-factor in AlphaFold)
                            b_factors = [atom.get_bfactor() for atom in residue]
                            plddt = np.mean(b_factors) if b_factors else 0
                            
                            # Try to get secondary structure
                            ss_features = np.zeros(3)  # [helix, sheet, coil]
                            try:
                                if (chain.id, residue.id) in dssp_dict:
                                    ss = dssp_dict[(chain.id, residue.id)][2]
                                    if ss in ['H', 'G', 'I']:  # Various helices
                                        ss_features[0] = 1
                                    elif ss in ['E', 'B']:  # Sheets
                                        ss_features[1] = 1
                                    else:  # Coils, turns, etc.
                                        ss_features[2] = 1
                                else:
                                    # Default to coil if not found
                                    ss_features[2] = 1
                            except:
                                # Default to coil if error
                                ss_features[2] = 1
                            
                            # Store features
                            aa_encoded = one_hot_encode(aa)
                            aa_group = get_aa_group_encoding(aa)
                            phys_props = get_physicochemical_props(aa)
                            
                            basic_features[res_id] = {
                                'aa_encoded': aa_encoded,
                                'aa_group': aa_group,
                                'phys_props': phys_props,
                                'plddt': plddt / 100.0,  # Normalize to 0-1
                                'ss_features': ss_features,
                                'chain': chain.id,
                                'residue_number': residue.id[1],
                                'amino_acid': aa
                            }
                            chain_residues.append(res_id)
                
                # Sort residues by number for each chain
                chain_residues.sort(key=lambda x: int(x.split('_')[1]))
                residue_list.extend(chain_residues)
        
        # Add context features
        enhanced_features = {}
        
        for i, res_id in enumerate(residue_list):
            # Base features
            current_features = basic_features[res_id]
            
            # Initialize window features
            window_aa = []
            window_group = []
            window_phys = []
            window_plddt = []
            window_ss = []
            
            # Get window around current residue
            for offset in range(-window_size, window_size + 1):
                idx = i + offset
                
                # Check if index is valid and residue is from same chain
                if (idx >= 0 and idx < len(residue_list) and 
                    residue_list[idx].split('_')[0] == res_id.split('_')[0]):
                    
                    neighbor_id = residue_list[idx]
                    neighbor = basic_features[neighbor_id]
                    
                    # Append features
                    window_aa.append(neighbor['aa_encoded'])
                    window_group.append(neighbor['aa_group'])
                    window_phys.append(neighbor['phys_props'])
                    window_plddt.append(neighbor['plddt'])
                    window_ss.append(neighbor['ss_features'])
                else:
                    # Padding for out-of-bounds
                    window_aa.append(np.zeros(20))
                    window_group.append(np.zeros(5))
                    window_phys.append(np.zeros(len(current_features['phys_props'])))
                    window_plddt.append(0.0)
                    window_ss.append(np.zeros(3))
            
            # Combine features
            flat_window_aa = np.concatenate(window_aa)
            flat_window_group = np.concatenate(window_group)
            flat_window_phys = np.concatenate(window_phys)
            window_plddt_array = np.array(window_plddt)
            flat_window_ss = np.concatenate(window_ss)
            
            # Create enhanced feature vector
            enhanced_features[res_id] = {
                **current_features,
                'window_aa': flat_window_aa,
                'window_group': flat_window_group,
                'window_phys': flat_window_phys,
                'window_plddt': window_plddt_array,
                'window_ss': flat_window_ss,
            }
        
        return enhanced_features
    except Exception as e:
        log(f"Error parsing CIF file {cif_path}: {e}")
        log(traceback.format_exc())
        return {}

# Function to extract binding site labels from P2Rank output
def extract_labels_from_p2rank(p2rank_path):
    try:
        # Read P2Rank residue predictions
        df = pd.read_csv(p2rank_path, sep=',', skipinitialspace=True)
        
        # Extract binding site predictions
        residue_labels = {}
        for _, row in df.iterrows():
            if 'pred_id' in df.columns:
                res_id = row['pred_id']
                predicted = row['predicted']
                residue_labels[res_id] = int(predicted)
        
        return residue_labels
    except Exception as e:
        log(f"Error parsing P2Rank file {p2rank_path}: {e}")
        log(traceback.format_exc())
        return {}

# Define focal loss for imbalanced classification
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
    return focal_loss_fixed

# Build transformer-based model for binding site prediction
def build_transformer_model(input_shape_dict):
    # Create inputs
    aa_input = Input(shape=(input_shape_dict['window_aa'],), name='aa_input')
    group_input = Input(shape=(input_shape_dict['window_group'],), name='group_input')
    phys_input = Input(shape=(input_shape_dict['window_phys'],), name='phys_input')
    plddt_input = Input(shape=(input_shape_dict['window_plddt'],), name='plddt_input')
    ss_input = Input(shape=(input_shape_dict['window_ss'],), name='ss_input')
    
    # Reshape inputs
    window_size = input_shape_dict['window_plddt']
    aa_features = 20
    group_features = 5
    phys_features = len(input_shape_dict['window_phys']) // window_size
    ss_features = 3
    
    # Reshape inputs for attention mechanisms
    reshaped_aa = Reshape((window_size, aa_features))(aa_input)
    reshaped_group = Reshape((window_size, group_features))(group_input)
    reshaped_phys = Reshape((window_size, phys_features))(phys_input)
    reshaped_plddt = Reshape((window_size, 1))(plddt_input)
    reshaped_ss = Reshape((window_size, ss_features))(ss_input)
    
    # Concatenate all inputs along feature dimension
    concat_input = Concatenate(axis=2)([
        reshaped_aa, reshaped_group, reshaped_phys, 
        reshaped_plddt, reshaped_ss
    ])
    
    # Apply layer normalization before attention
    x = LayerNormalization(epsilon=1e-6)(concat_input)
    
    # Apply attention mechanism (self-attention)
    # Using Keras-compatible methods instead of direct TF operations
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=8
    )(x, x)
    
    # Skip connection and normalization
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Position-wise feed-forward network
    ffn = Dense(128, activation='relu')(x)
    ffn = Dropout(0.4)(ffn)
    ffn = Dense(aa_features + group_features + phys_features + 1 + ss_features)(ffn)
    
    # Skip connection and normalization
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Extract both global and local features
    conv = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    conv = LayerNormalization()(conv)
    
    # Global representations
    global_avg = GlobalAveragePooling1D()(x)
    global_max = GlobalMaxPooling1D()(x)
    
    # Central residue focus - extract features from the middle position
    middle_idx = window_size // 2
    central_features = tf.keras.layers.Lambda(
        lambda t: t[:, middle_idx, :]
    )(x)
    
    # Combine global and local representations
    concat_features = Concatenate()([
        global_avg, global_max, central_features,
        Flatten()(conv)
    ])
    
    # Final MLP classifier
    x = Dense(256)(concat_features)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128)(x)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(
        inputs=[aa_input, group_input, phys_input, plddt_input, ss_input], 
        outputs=output
    )
    
    # Compile model with focal loss
    model.compile(
        optimizer=Adam(learning_rate=0.0005, weight_decay=1e-5),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model

# Process all proteins
all_window_aa = []
all_window_group = []
all_window_phys = []
all_window_plddt = []
all_window_ss = []
all_labels = []
protein_info = []
protein_groups = []  # For cross-validation

# Get all CIF files
cif_files = list(Path(alphafold_dir).glob("*.cif"))
log(f"Found {len(cif_files)} AlphaFold CIF files")

processed = 0
start_time = time.time()
window_size = 5  # Window size for sequence context

for idx, cif_file in enumerate(cif_files):
    cif_name = cif_file.name
    # Extract protein ID for grouping
    protein_id = cif_name.split('-')[1] if '-' in cif_name else cif_name.split('.')[0]
    p2rank_file = Path(p2rank_dir) / f"{cif_name}_residues.csv"
    
    # Check if P2Rank file exists
    if not p2rank_file.exists():
        log(f"P2Rank file not found for {cif_name}, skipping")
        continue
    
    # Show progress
    if idx % 10 == 0:
        elapsed = time.time() - start_time
        per_protein = elapsed / (idx + 1) if idx > 0 else 0
        remaining = per_protein * (len(cif_files) - idx - 1)
        log(f"Processing {idx+1}/{len(cif_files)}: {cif_name} (Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s)")
    
    try:
        # Extract features and labels
        residue_features = extract_features_from_cif(cif_file, window_size)
        residue_labels = extract_labels_from_p2rank(p2rank_file)
        
        # Match features with labels
        matched_count = 0
        binding_sites = 0
        
        for res_id, features in residue_features.items():
            if res_id in residue_labels:
                # Get window features
                window_aa = features['window_aa']
                window_group = features['window_group']
                window_phys = features['window_phys']
                window_plddt = features['window_plddt']
                window_ss = features['window_ss']
                label = residue_labels[res_id]
                
                # Append to lists
                all_window_aa.append(window_aa)
                all_window_group.append(window_group)
                all_window_phys.append(window_phys)
                all_window_plddt.append(window_plddt)
                all_window_ss.append(window_ss)
                all_labels.append(label)
                protein_groups.append(protein_id)  # Save protein ID for group-based CV
                
                # Store metadata
                protein_info.append({
                    'protein': cif_name,
                    'protein_id': protein_id,
                    'chain': features['chain'],
                    'residue_number': features['residue_number'],
                    'amino_acid': features['amino_acid'],
                    'binding_site': label
                })
                
                matched_count += 1
                binding_sites += label
        
        if matched_count > 0:
            log(f"  Matched {matched_count} residues, {binding_sites} binding sites")
            processed += 1
        else:
            log(f"  No matches found for {cif_name}")
    
    except Exception as e:
        log(f"Error processing {cif_name}: {e}")
        log(traceback.format_exc())

log(f"Processed {processed} proteins successfully")
log(f"Total residues: {len(all_labels)}")
log(f"Binding sites identified: {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.2f}%)")

# Convert to numpy arrays
X_window_aa = np.array(all_window_aa)
X_window_group = np.array(all_window_group)
X_window_phys = np.array(all_window_phys)
X_window_plddt = np.array(all_window_plddt)
X_window_ss = np.array(all_window_ss)
y = np.array(all_labels)
protein_groups = np.array(protein_groups)

# Save protein info
pd.DataFrame(protein_info).to_csv(os.path.join(output_dir, "protein_residue_info.csv"), index=False)

# Save features and labels
np.save(os.path.join(output_dir, "window_aa.npy"), X_window_aa)
np.save(os.path.join(output_dir, "window_group.npy"), X_window_group)
np.save(os.path.join(output_dir, "window_phys.npy"), X_window_phys)
np.save(os.path.join(output_dir, "window_plddt.npy"), X_window_plddt)
np.save(os.path.join(output_dir, "window_ss.npy"), X_window_ss)
np.save(os.path.join(output_dir, "labels.npy"), y)
np.save(os.path.join(output_dir, "protein_groups.npy"), protein_groups)

# Define cross-validation strategy
log("Setting up protein-based cross-validation")
n_splits = 5
group_kfold = GroupKFold(n_splits=n_splits)

# Store CV results
cv_scores = []
cv_roc_auc = []
cv_pr_auc = []

# Feature shape dictionary
input_shape_dict = {
    'window_aa': X_window_aa.shape[1],
    'window_group': X_window_group.shape[1],
    'window_phys': X_window_phys.shape[1],
    'window_plddt': X_window_plddt.shape[1],
    'window_ss': X_window_ss.shape[1]
}

# Prepare for cross-validation
fold_predictions = []
fold_true_values = []
fold_indices = []
fold_proteins = []

# Class weights for focal loss are built into the loss function
log("Starting cross-validation training...")

for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_window_aa, y, groups=protein_groups)):
    log(f"\nFold {fold+1}/{n_splits}")
    
    # Split train into train and validation
    train_proteins = protein_groups[train_idx]
    train_data = X_window_aa[train_idx]
    train_labels = y[train_idx]
    
    # Use GroupKFold again to create validation set from training set
    inner_split = next(GroupKFold(n_splits=4).split(train_data, train_labels, groups=train_proteins))
    inner_train_idx, val_idx = inner_split
    
    # Map indices back to original dataset
    final_train_idx = train_idx[inner_train_idx]
    final_val_idx = train_idx[val_idx]
    
    # Create data splits
    X_train = {
        'aa_input': X_window_aa[final_train_idx],
        'group_input': X_window_group[final_train_idx],
        'phys_input': X_window_phys[final_train_idx],
        'plddt_input': X_window_plddt[final_train_idx],
        'ss_input': X_window_ss[final_train_idx]
    }
    
    X_val = {
        'aa_input': X_window_aa[final_val_idx],
        'group_input': X_window_group[final_val_idx],
        'phys_input': X_window_phys[final_val_idx],
        'plddt_input': X_window_plddt[final_val_idx],
        'ss_input': X_window_ss[final_val_idx]
    }
    
    X_test = {
        'aa_input': X_window_aa[test_idx],
        'group_input': X_window_group[test_idx],
        'phys_input': X_window_phys[test_idx],
        'plddt_input': X_window_plddt[test_idx],
        'ss_input': X_window_ss[test_idx]
    }
    
    y_train = y[final_train_idx]
    y_val = y[final_val_idx]
    y_test = y[test_idx]
    
    # Log dataset sizes
    log(f"Training set: {len(y_train)} examples, {sum(y_train)} positives ({sum(y_train)/len(y_train)*100:.2f}%)")
    log(f"Validation set: {len(y_val)} examples, {sum(y_val)} positives ({sum(y_val)/len(y_val)*100:.2f}%)")
    log(f"Test set: {len(y_test)} examples, {sum(y_test)} positives ({sum(y_test)/len(y_test)*100:.2f}%)")
    
    # Build model
    log(f"Building model for fold {fold+1}...")
    model = build_transformer_model(input_shape_dict)
    
    # Save unique fold proteins for analysis
    fold_proteins.append({
        'fold': fold+1,
        'test_proteins': np.unique(protein_groups[test_idx]).tolist()
    })
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, f'model_fold{fold+1}.keras'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # Train model
    log(f"Training fold {fold+1}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=2
    )
    
    # Plot training history for this fold
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Fold {fold+1} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Fold {fold+1} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold{fold+1}_history.png"))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log(f"Fold {fold+1} test accuracy: {test_acc:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc_score = auc(fpr, tpr)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc_score = average_precision_score(y_test, y_pred)
    
    # Find optimal threshold (maximize F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = _[optimal_idx]
    log(f"Fold {fold+1} optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    
    # Convert to binary predictions using optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    try:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        log(f"Fold {fold+1} sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}")
    except:
        log(f"Warning: Could not calculate all confusion matrix metrics for fold {fold+1}")
    
    # Save fold metrics
    cv_scores.append(test_acc)
    cv_roc_auc.append(roc_auc_score)
    cv_pr_auc.append(pr_auc_score)
    
    # Store predictions for analysis
    fold_predictions.append(y_pred)
    fold_true_values.append(y_test)
    fold_indices.append(test_idx)
    
    # Plot ROC and PR curves for this fold
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold+1} ROC Curve (AUC = {roc_auc_score:.3f})')
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold+1} PR Curve (AP = {pr_auc_score:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold{fold+1}_curves.png"))
    
    # Clear plot memory
    plt.close('all')
    
    # Save fold results
    fold_results = {
        'test_accuracy': float(test_acc),
        'roc_auc': float(roc_auc_score),
        'pr_auc': float(pr_auc_score),
        'optimal_threshold': float(optimal_threshold),
        'early_stopping_epoch': len(history.history['loss']),
        'history': {k: [float(val) for val in v] for k, v in history.history.items()}
    }
    
    with open(os.path.join(output_dir, f"fold{fold+1}_results.json"), 'w') as f:
        json.dump(fold_results, f, indent=2)

# Print average cross-validation results
log("\nCross-validation results:")
log(f"Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
log(f"Mean ROC AUC: {np.mean(cv_roc_auc):.4f} ± {np.std(cv_roc_auc):.4f}")
log(f"Mean PR AUC: {np.mean(cv_pr_auc):.4f} ± {np.std(cv_pr_auc):.4f}")

# Train final model on all data
log("\nTraining final model on all data...")

# Prepare data
X_all = {
    'aa_input': X_window_aa,
    'group_input': X_window_group,
    'phys_input': X_window_phys,
    'plddt_input': X_window_plddt,
    'ss_input': X_window_ss
}

# Split for final validation
train_idx, val_idx = next(GroupKFold(n_splits=5).split(X_window_aa, y, groups=protein_groups))

X_train_final = {k: v[train_idx] for k, v in X_all.items()}
X_val_final = {k: v[val_idx] for k, v in X_all.items()}
y_train_final = y[train_idx]
y_val_final = y[val_idx]

# Build final model
final_model = build_transformer_model(input_shape_dict)

# Callbacks for final model
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(output_dir, 'final_model.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Train final model
history = final_model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val_final, y_val_final),
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=2
)

# Save final model
final_model.save(os.path.join(output_dir, "binding_site_model.keras"))
log("Final model saved")

# Plot training history for final model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Final Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Final Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_model_history.png"))

# Evaluate final model
val_loss, val_acc = final_model.evaluate(X_val_final, y_val_final, verbose=0)
log(f"Final model validation accuracy: {val_acc:.4f}")

# Make predictions on validation set
y_pred = final_model.predict(X_val_final).flatten()

# ROC curve
fpr, tpr, _ = roc_curve(y_val_final, y_pred)
roc_auc_score = auc(fpr, tpr)

# PR curve
precision, recall, thresholds = precision_recall_curve(y_val_final, y_pred)
pr_auc_score = average_precision_score(y_val_final, y_pred)

# Find optimal threshold (maximize F1 score)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
log(f"Final model optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")

# Convert to binary predictions using optimal threshold
y_pred_binary = (y_pred >= optimal_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_val_final, y_pred_binary)
try:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    log(f"Final model sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}")
except:
    log("Warning: Could not calculate all confusion matrix metrics for final model")

# Save final model results
final_results = {
    'validation_accuracy': float(val_acc),
    'roc_auc': float(roc_auc_score),
    'pr_auc': float(pr_auc_score),
    'optimal_threshold': float(optimal_threshold),
    'early_stopping_epoch': len(history.history['loss']),
    'cross_validation_mean_accuracy': float(np.mean(cv_scores)),
    'cross_validation_std_accuracy': float(np.std(cv_scores)),
    'cross_validation_mean_roc_auc': float(np.mean(cv_roc_auc)),
    'cross_validation_std_roc_auc': float(np.std(cv_roc_auc)),
    'cross_validation_mean_pr_auc': float(np.mean(cv_pr_auc)),
    'cross_validation_std_pr_auc': float(np.std(cv_pr_auc)),
    'history': {k: [float(val) for val in v] for k, v in history.history.items()},
    'fold_proteins': fold_proteins
}

with open(os.path.join(output_dir, "final_results.json"), 'w') as f:
    json.dump(final_results, f, indent=2)

# Plot ROC and PR curves for final model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Final Model ROC Curve (AUC = {roc_auc_score:.3f})')

plt.subplot(1, 2, 2)
plt.plot(recall, precision, lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Final Model PR Curve (AP = {pr_auc_score:.3f})')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_model_curves.png"))

# Protein-level analysis
log("\nAnalyzing performance at the protein level...")
protein_df = pd.DataFrame(protein_info)

# Get predictions for validation set proteins
protein_metrics = []
protein_predictions = []

val_proteins = np.unique(protein_groups[val_idx])
for protein in val_proteins:
    # Get residues for this protein
    protein_residue_mask = protein_df['protein_id'] == protein
    protein_val_indices = [i for i, idx in enumerate(val_idx) if protein_groups[idx] == protein]
    
    if len(protein_val_indices) > 0:
        # Get predictions
        protein_y_true = y_val_final[protein_val_indices]
        protein_y_pred = y_pred[protein_val_indices]
        protein_y_binary = y_pred_binary[protein_val_indices]
        
        # Calculate metrics
        accuracy = np.mean(protein_y_binary == protein_y_true)
        
        try:
            roc_auc_val = roc_auc_score(protein_y_true, protein_y_pred)
        except:
            roc_auc_val = np.nan
            
        try:
            pr_auc_val = average_precision_score(protein_y_true, protein_y_pred)
        except:
            pr_auc_val = np.nan
        
        try:
            cm = confusion_matrix(protein_y_true, protein_y_binary)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        except:
            sensitivity = np.nan
            specificity = np.nan
        
        # Add to protein metrics
        protein_metrics.append({
            'protein_id': protein,
            'n_residues': len(protein_val_indices),
            'n_binding_sites': np.sum(protein_y_true),
            'binding_site_ratio': np.mean(protein_y_true),
            'accuracy': accuracy,
            'roc_auc': roc_auc_val,
            'pr_auc': pr_auc_val,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
        
        # Add to protein predictions for visualization
        protein_residue_info = protein_df[protein_residue_mask].copy()
        if len(protein_residue_info) > 0:
            residue_indices = {
                (row['chain'], row['residue_number']): i 
                for i, (_, row) in enumerate(protein_residue_info.iterrows())
            }
            
            for i, val_idx_pos in enumerate(protein_val_indices):
                val_residue_idx = val_idx[val_idx_pos]
                chain = protein_info[val_residue_idx]['chain']
                residue_number = protein_info[val_residue_idx]['residue_number']
                
                if (chain, residue_number) in residue_indices:
                    row_idx = residue_indices[(chain, residue_number)]
                    protein_residue_info.iloc[row_idx, protein_residue_info.columns.get_loc('pred_score')] = protein_y_pred[i]
                    protein_residue_info.iloc[row_idx, protein_residue_info.columns.get_loc('pred_class')] = protein_y_binary[i]
            
            protein_predictions.append(protein_residue_info)

# Save protein-level metrics
pd.DataFrame(protein_metrics).to_csv(os.path.join(output_dir, "protein_level_metrics.csv"), index=False)

# Generate summary visualizations
log("\nGenerating summary visualizations...")

# Plot metrics distributions
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(protein_metrics, x='accuracy', kde=True)
plt.title('Protein Accuracy Distribution')

plt.subplot(2, 2, 2)
sns.histplot(protein_metrics, x='roc_auc', kde=True)
plt.title('Protein ROC-AUC Distribution')

plt.subplot(2, 2, 3)
sns.histplot(protein_metrics, x='pr_auc', kde=True)
plt.title('Protein PR-AUC Distribution')

plt.subplot(2, 2, 4)
sns.scatterplot(data=pd.DataFrame(protein_metrics), x='binding_site_ratio', y='accuracy', hue='n_residues', size='n_residues')
plt.title('Accuracy vs Binding Site Ratio')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "protein_metrics_distribution.png"))

# Generate feature importance analysis
log("\nGenerating feature importance analysis...")

# Feature names
feature_names = {
    'aa_input': [f'aa_{i}' for i in range(input_shape_dict['window_aa'])],
    'group_input': [f'group_{i}' for i in range(input_shape_dict['window_group'])],
    'phys_input': [f'phys_{i}' for i in range(input_shape_dict['window_phys'])],
    'plddt_input': [f'plddt_{i}' for i in range(input_shape_dict['window_plddt'])],
    'ss_input': [f'ss_{i}' for i in range(input_shape_dict['window_ss'])]
}

# Simple permutation importance for key features
important_features = []

log("Training and evaluation complete!")
log(f"Results saved to {output_dir}/")

# Create a simple prediction script for future use
prediction_script = f"""#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import MMCIFParser
import tensorflow as tf
from tensorflow.keras.models import load_model

# Custom objects for the model
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
    return focal_loss_fixed

# Load model
model_path = os.path.join("{output_dir}", "binding_site_model.keras")
model = load_model(model_path, custom_objects={{'focal_loss_fixed': focal_loss()}})

# Load optimal threshold
with open(os.path.join("{output_dir}", "final_results.json"), 'r') as f:
    results = json.load(f)
    optimal_threshold = results['optimal_threshold']

# Function to extract features from AlphaFold CIF files
# [Copy the feature extraction functions from the training script]

def predict_binding_sites(cif_path, output_path=None):
    # Extract features
    residue_features = extract_features_from_cif(cif_path)
    
    # Prepare data for prediction
    window_aa = []
    window_group = []
    window_phys = []
    window_plddt = []
    window_ss = []
    residue_info = []
    
    for res_id, features in residue_features.items():
        window_aa.append(features['window_aa'])
        window_group.append(features['window_group'])
        window_phys.append(features['window_phys'])
        window_plddt.append(features['window_plddt'])
        window_ss.append(features['window_ss'])
        
        residue_info.append({{
            'res_id': res_id,
            'chain': features['chain'],
            'residue_number': features['residue_number'],
            'amino_acid': features['amino_acid']
        }})
    
    # Convert to numpy arrays
    X_test = {{
        'aa_input': np.array(window_aa),
        'group_input': np.array(window_group),
        'phys_input': np.array(window_phys),
        'plddt_input': np.array(window_plddt),
        'ss_input': np.array(window_ss)
    }}
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Convert to binary using optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Create results dataframe
    results_df = pd.DataFrame(residue_info)
    results_df['binding_score'] = y_pred
    results_df['binding_site'] = y_pred_binary
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict protein binding sites')
    parser.add_argument('cif_path', help='Path to AlphaFold CIF file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    
    args = parser.parse_args()
    
    predictions = predict_binding_sites(args.cif_path, args.output)
    print(f"Predicted {len(predictions)} residues, {predictions['binding_site'].sum()} binding sites")
"""

# Save prediction script
with open(os.path.join(output_dir, "predict_binding_sites.py"), 'w') as f:
    f.write(prediction_script)

log("Prediction script saved for future use")
EOF

# Run the Python script
echo "[$(date)] Starting advanced binding site model training"
python train_advanced_model.py
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "[$(date)] Training failed with exit code $TRAINING_EXIT_CODE"
    exit 1
fi

# Save training metadata
echo "Saving training metadata..."
cat > "$OUTPUT_DIR/training_metadata.txt" << EOL
Job ID: ${SLURM_JOB_ID:-N/A}
Training started: $(date -d @$(($(date +%s) - SECONDS)))
Training completed: $(date)
Duration: $((SECONDS/3600))h $((SECONDS%3600/60))m $((SECONDS%60))s

Advanced Model Features:
1. Enhanced Feature Engineering:
   - One-hot amino acid encoding + amino acid group encoding
   - Expanded physicochemical properties (hydrophobicity, volume, charge, polarity, surface exposure)
   - Secondary structure features when available
   - pLDDT confidence scores from AlphaFold
   - Window-based sequence context (±5 residues)

2. Transformer-Based Architecture:
   - Self-attention mechanism for capturing relationships between residues
   - Multi-head attention to learn different aspects of binding patterns
   - Skip connections and layer normalization for stable training
   - Local and global feature extraction

3. Advanced Training Approach:
   - Protein-based cross-validation (GroupKFold)
   - Focal loss for handling class imbalance
   - Learning rate scheduling with weight decay
   - Early stopping with patience and model checkpointing

4. Comprehensive Evaluation:
   - Per-protein performance metrics
   - ROC and PR curve analysis
   - Optimal threshold selection
   - Performance visualization across protein types
EOL

# Copy log file to results directory
if [ -n "$SLURM_JOB_ID" ]; then
    LOG_FILE="binding_site_adv_${SLURM_JOB_ID}.log"
    if [ -f "$LOG_FILE" ]; then
        cp "$LOG_FILE" "$OUTPUT_DIR/"
        echo "Log file copied to results directory."
    fi
fi

echo "[$(date)] Job completed"
echo "Results saved in $OUTPUT_DIR"
echo "Duration: $((SECONDS/3600))h $((SECONDS%3600/60))m $((SECONDS%60))s"
