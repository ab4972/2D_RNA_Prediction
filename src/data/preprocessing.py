"""
Preprocessing utilities for RNA structure prediction data.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import logging
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.rna_embeddings import RNAFM
from src.utils.vienna_rna import get_basepair_probabilities, dotbracket_to_matrix

logger = logging.getLogger(__name__)

class RNADataset(Dataset):
    """
    Dataset for RNA structure prediction.
    """
    def __init__(self, data_path, cache_dir=None, max_seq_len=512, generate_features=True):
        """
        Initialize the RNA dataset.
        
        Args:
            data_path (str): Path to the dataset CSV file
            cache_dir (str): Directory to cache generated features
            max_seq_len (int): Maximum sequence length to include
            generate_features (bool): Whether to generate features on initialization
        """
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        
        # Load the dataset
        logger.info(f"Loading dataset from {data_path}")
        self.df = pd.read_csv(data_path)
        
        # Filter by sequence length
        orig_len = len(self.df)
        self.df = self.df[self.df['sequence'].apply(len) <= max_seq_len]
        logger.info(f"Filtered dataset from {orig_len} to {len(self.df)} samples (max length: {max_seq_len})")
        
        # Initialize feature caching
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Generate features if requested
        if generate_features:
            self.generate_and_cache_features()
            
    def generate_and_cache_features(self):
        """Generate and cache RNA-FM embeddings and ViennaRNA features."""
        # Initialize RNA-FM model
        rna_fm = RNAFM()
        
        # Track processed items
        self.processed_indices = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Generating features"):
            sequence = row['sequence']
            sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
            
            if len(sequence) == 0:
                continue
                
            cache_path = None
            if self.cache_dir:
                # Create cache filename
                cache_path = os.path.join(self.cache_dir, f"features_{idx}.npz")
                
                # Skip if cache exists
                if os.path.exists(cache_path):
                    self.processed_indices.append(idx)
                    continue
            
            try:
                # Generate RNA-FM embeddings
                embeddings = rna_fm.get_embeddings(sequence)
                
                # Generate ViennaRNA basepair probabilities
                bp_probs = get_basepair_probabilities(sequence)
                
                # Generate target matrix from dot-bracket (if available)
                target = None
                if 'dot_bracket' in row:
                    target = dotbracket_to_matrix(row['dot_bracket'])
                
                # Save to cache if enabled
                if cache_path:
                    data_to_save = {
                        'embeddings': embeddings,
                        'bp_probs': bp_probs
                    }
                    if target is not None:
                        data_to_save['target'] = target
                        
                    np.savez_compressed(cache_path, **data_to_save)
                
                self.processed_indices.append(idx)
                
            except Exception as e:
                logger.error(f"Error processing sequence {idx}: {e}")
        
        # Filter dataframe to only include processed samples
        self.df = self.df.loc[self.processed_indices].reset_index(drop=True)
        logger.info(f"Final dataset contains {len(self.df)} sequences after feature generation")

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        """Get a data item with features."""
        row = self.df.iloc[idx]
        sequence = row['sequence']
        
        # Load cached features if available
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"features_{self.processed_indices[idx]}.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                embeddings = data['embeddings']
                bp_probs = data['bp_probs']
                target = data.get('target', None)
                
                return {
                    'sequence': sequence,
                    'embeddings': torch.tensor(embeddings, dtype=torch.float32),
                    'bp_probs': torch.tensor(bp_probs, dtype=torch.float32),
                    'target': torch.tensor(target, dtype=torch.float32) if target is not None else None,
                    'length': len(sequence)
                }
        
        # If no cache, this should not happen if generate_features=True
        raise ValueError("Features not found in cache. Run with generate_features=True first.")

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Args:
        batch (list): List of dataset items
        
    Returns:
        dict: Batched tensors
    """
    # Get max sequence length in batch
    max_len = max([item['length'] for item in batch])
    
    # Prepare batch data
    sequences = [item['sequence'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Pad embeddings
    padded_embeddings = []
    for item in batch:
        emb = item['embeddings']
        padding = torch.zeros((max_len - emb.shape[0], emb.shape[1]), dtype=torch.float32)
        padded_embeddings.append(torch.cat([emb, padding], dim=0))
    embeddings = torch.stack(padded_embeddings)
    
    # Pad bp_probs
    padded_bp_probs = []
    for item in batch:
        bp = item['bp_probs']
        padding_rows = torch.zeros((max_len - bp.shape[0], bp.shape[1]), dtype=torch.float32)
        padded_bp = torch.cat([bp, padding_rows], dim=0)
        padding_cols = torch.zeros((max_len, max_len - bp.shape[1]), dtype=torch.float32)
        padded_bp = torch.cat([padded_bp, padding_cols], dim=1)
        padded_bp_probs.append(padded_bp)
    bp_probs = torch.stack(padded_bp_probs)
    
    # Pad targets (if available)
    targets = None
    if batch[0]['target'] is not None:
        padded_targets = []
        for item in batch:
            tgt = item['target']
            padding_rows = torch.zeros((max_len - tgt.shape[0], tgt.shape[1]), dtype=torch.float32)
            padded_tgt = torch.cat([tgt, padding_rows], dim=0)
            padding_cols = torch.zeros((max_len, max_len - tgt.shape[1]), dtype=torch.float32)
            padded_tgt = torch.cat([padded_tgt, padding_cols], dim=1)
            padded_targets.append(padded_tgt)
        targets = torch.stack(padded_targets)
    
    return {
        'sequences': sequences,
        'embeddings': embeddings,
        'bp_probs': bp_probs,
        'targets': targets,
        'lengths': lengths
    }

def get_dataloaders(data_dir, cache_dir='data/cache', batch_size=16, 
                   max_seq_len=512, num_workers=4, val_split=False, test_split=False):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Path to data directory containing CSV files or a main CSV file
        cache_dir (str): Directory to cache features
        batch_size (int): Batch size
        max_seq_len (int): Maximum sequence length
        num_workers (int): Number of workers for data loading
        val_split (bool): Whether to include validation data
        test_split (bool): Whether to include test data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoaders for each split
    """
    # Determine paths for data files
    if os.path.isdir(data_dir):
        # If data_dir is a directory, assume it contains train.csv, val.csv, etc.
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv") if val_split else None
        test_path = os.path.join(data_dir, "test.csv") if test_split else None
    else:
        # If data_dir is a file, use it as the train path
        train_path = data_dir
        val_path = None
        test_path = None
    
    # Create data loaders
    train_cache = os.path.join(cache_dir, 'train')
    train_dataset = RNADataset(train_path, cache_dir=train_cache, max_seq_len=max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Validation data (if requested)
    val_loader = None
    if val_split and val_path and os.path.exists(val_path):
        val_cache = os.path.join(cache_dir, 'val')
        val_dataset = RNADataset(val_path, cache_dir=val_cache, max_seq_len=max_seq_len)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    # Test data (if requested)
    test_loader = None
    if test_split and test_path and os.path.exists(test_path):
        test_cache = os.path.join(cache_dir, 'test')
        test_dataset = RNADataset(test_path, cache_dir=test_cache, max_seq_len=max_seq_len)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader 