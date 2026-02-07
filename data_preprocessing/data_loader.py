import torch
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter

from data_preprocessing.data_preprocess import preprocess_dataset, pad_sequences, balance_by_augmentation
from data_preprocessing.doore_preprocess import dooreLoader
from data_preprocessing.opportunity_preprocess import opportunityLoader
from data_preprocessing.casas_preprocess import casasLoader
from data_preprocessing.aras_preprocess import arasLoader
from data_preprocessing.openpack_preprocess import openpackLoader

import logging

# ========== Main Functions ==========

def prepare_kfold_datasets(args):
    """
    Prepare K-Fold cross-validation datasets from raw data.
    
    Steps:
        1. Load dataset using appropriate loader
        2. Filter by minimum sample count per class
        3. Normalize and relabel
        4. Split into K folds and save to disk
        5. Save metadata (label_map, input_channels)
    
    Args:
        args: Arguments containing dataset name, timespan, min_seq, k_fold, etc.
    
    Returns:
        tuple: (label_map, input_channels)
    """
    # 1. Load dataset
    if args.dataset == 'doore':
        dataset_list = dooreLoader('data/doore/*.csv', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'opportunity':
        dataset_list = opportunityLoader('data/opportunity/*.dat', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'casas':
        dataset_list = casasLoader('data/casas/*.txt', min_seq=args.min_seq)
    elif args.dataset == 'aras':
        dataset_list = arasLoader('data/aras/HouseA/*.txt', timespan=args.timespan, min_seq=args.min_seq)
    elif args.dataset == 'openpack':
        dataset_list = openpackLoader('data/openpack/*.csv', timespan=args.timespan, min_seq=args.min_seq)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 2. Filter by class count
    if args.min_samples > 0:
        raw_labels = [seq.label for seq in dataset_list]
        counts = Counter(raw_labels)
        valid_labels = {label for label, count in counts.items() if count >= args.min_samples}
        dataset_list = [seq for seq in dataset_list if seq.label in valid_labels]
        logging.info(f"[Filter] Classes ≥ {args.min_samples}: {sorted(valid_labels)}")

        # Re-count filtered data
        filtered_counts = Counter(seq.label for seq in dataset_list)
        total_sequences_filtered = sum(filtered_counts.values())
        total_pointers_filtered = sum(seq.length for seq in dataset_list)
        logging.info(f"[After Filter] Total sequences: {total_sequences_filtered}")
        logging.info(f"[After Filter] Total data points: {total_pointers_filtered}")
        for label in sorted(filtered_counts.keys()):
            count = filtered_counts[label]
            total_points = sum(seq.length for seq in dataset_list if seq.label == label)
            logging.info(f"  Activity {label}: {count} sequences, {total_points} data points")

    # 3. Normalize + Relabel (No augment, No padding)
    normalized_seqs, label_list, label_map = preprocess_dataset(
        dataset_list,
        padding_type=None,
        augment_method=None
    )

    labels_tensor = torch.tensor(label_list, dtype=torch.long)

    # 4. K-Fold Stratified Split + Save
    split_and_save_kfold_data(
        normalized_seqs,
        labels_tensor,
        k=args.k_fold,
        output_dir=f"fold_data/{args.dataset}"
    )

    # 5. Save meta information
    with open(f"fold_data/{args.dataset}/meta.pkl", "wb") as f:
        pickle.dump({
            "label_map": label_map,
            "input_channels": normalized_seqs[0].shape[1]
        }, f)    

    return label_map, normalized_seqs[0].shape[1]

def load_fold_data(fold_path, args, mode='train'):
    """
    Load a specific fold from saved pickle file and prepare DataLoaders.
    
    Steps:
        1. Load fold data (train/test split)
        2. Split validation from training set
        3. Apply augmentation (training only)
        4. Apply padding to all sets
        5. Create PyTorch DataLoaders
    
    Args:
        fold_path: Path to fold pickle file
        args: Arguments containing batch_size, padding, augment settings
        mode: 'train' or 'test' (augmentation only applied in train mode)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load fold data from saved pickle file
    with open(fold_path, "rb") as f:
        fold_data = pickle.load(f)

    train_data = fold_data['train_data']
    test_data = fold_data['test_data']
    train_labels = fold_data['train_labels']
    test_labels = fold_data['test_labels']

    # Split validation from train set
    val_ratio_adjusted = args.valid_ratio
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels,
        test_size=val_ratio_adjusted,
        stratify=train_labels,
        random_state=42
    )

    # Apply augmentation on training data only
    if mode == 'train' and args.augment:
        logging.info(f"[Augment] Method: {args.aug_method}")
        train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=args.aug_method)

    # Apply padding to all datasets
    logging.info("[Padding] Applying padding to fold data...")
    train_tensor, train_lengths = pad_sequences(train_data, padding_type=args.padding)
    val_tensor, val_lengths = pad_sequences(val_data, padding_type=args.padding)
    test_tensor, test_lengths = pad_sequences(test_data, padding_type=args.padding)

    logging.info(f"[Train] Tensor shape: {train_tensor.shape}")
    # Construct DataLoaders
    train_loader = DataLoader(TensorDataset(train_tensor, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_labels), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size=args.batch_size)

    logging.info(f"[Fold Loaded] Train: {len(train_tensor)}, Val: {len(val_tensor)}, Test: {len(test_tensor)}")

    return train_loader, val_loader, test_loader

# ========== Helper Functions ==========

def split_and_save_kfold_data(sequences, labels, k, output_dir, prefix="fold"):
    """
    Split data into K folds using Stratified K-Fold and save each fold to disk.
    
    Args:
        sequences: List of normalized sequences
        labels: Tensor of labels
        k: Number of folds
        output_dir: Directory to save fold files
        prefix: Filename prefix (default: "fold")
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx, (train_index, test_index) in enumerate(skf.split(sequences, labels)):
        train_data = [sequences[i] for i in train_index]
        test_data = [sequences[i] for i in test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        fold_data = {
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
        }

        with open(os.path.join(output_dir, f"{prefix}_{fold_idx}.pkl"), "wb") as f:
            pickle.dump(fold_data, f)

        logging.info(f"[Fold {fold_idx}] Saved: {len(train_data)} train / {len(test_data)} test")

    logging.info(f"[KFold Split] {k} folds saved to {output_dir}")

def combine_train_val(train_loader, val_loader, batch_size, shuffle=True, padding_type='mean'):
    """
    Combine train and validation loaders into a single loader for final retraining.
    
    Used during the final model training phase to maximize training data.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        batch_size: Batch size for combined loader
        shuffle: Whether to shuffle combined data
        padding_type: Padding type for sequences ('mean' or 'max')
    
    Returns:
        DataLoader: Combined train+val data loader
    """
    train_tensors, train_labels = zip(*[(x, y) for x, y in train_loader.dataset])
    val_tensors, val_labels = zip(*[(x, y) for x, y in val_loader.dataset])

    all_sequences = list(train_tensors) + list(val_tensors)
    all_labels = list(train_labels) + list(val_labels)

    padded_tensor, _ = pad_sequences(all_sequences, padding_type)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)

    dataset = TensorDataset(padded_tensor, label_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
