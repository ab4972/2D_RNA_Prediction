"""
RNA dataset loading and preprocessing utilities.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from src.utils.rna_embeddings import RNAFM

logger = logging.getLogger(__name__)

class RNADataset(Dataset):
    """Dataset for RNA sequences and their secondary structures."""
    
    def __init__(self, data_path, split='train', max_seq_len=512, 
                 use_embeddings=True, embedding_cache_dir=None, 
                 embedding_model=None):
        """
        Initialize the RNA dataset.
        
        Args:
            data_path (str): Path to the dataset directory or CSV file
            split (str): Data split ('train', 'val', or 'test')
            max_seq_len (int): Maximum sequence length to use
            use_embeddings (bool): Whether to use RNA-FM embeddings
            embedding_cache_dir (str): Directory to cache embeddings
            embedding_model: Pre-loaded RNA embedding model
        """
        self.max_seq_len = max_seq_len
        self.use_embeddings = use_embeddings
        self.embedding_cache_dir = embedding_cache_dir
        
        # Load the dataset
        data_path = Path(data_path)
        if data_path.is_dir():
            csv_path = data_path / f"{split}.csv"
        else:
            csv_path = data_path
            
        logger.info(f"Loading RNA data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Initialize embedding model if needed
        if use_embeddings:
            if embedding_model is None:
                logger.info("Initializing RNA embedding model")
                self.embedding_model = RNAFM()
            else:
                self.embedding_model = embedding_model
            
            # Create cache directory if it doesn't exist
            if embedding_cache_dir:
                os.makedirs(embedding_cache_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.df.iloc[idx]
        
        # Get sequence and secondary structure
        sequence = item['sequence']
        structure = item['secondary_structure']
        
        # Ensure they're not too long
        if len(sequence) > self.max_seq_len:
            start_idx = np.random.randint(0, len(sequence) - self.max_seq_len + 1)
            sequence = sequence[start_idx:start_idx + self.max_seq_len]
            structure = structure[start_idx:start_idx + self.max_seq_len]
        
        # Create one-hot encoding for sequence
        seq_one_hot = self._sequence_to_one_hot(sequence)
        
        # Convert structure string to labels/matrix
        contact_map = self._structure_to_contact_map(structure)
        
        # Get embeddings if needed
        if self.use_embeddings:
            embeddings = self._get_embeddings(sequence, item['id'])
        else:
            embeddings = None
        
        # Return appropriate data
        result = {
            'id': item['id'],
            'sequence': sequence,
            'structure': structure,
            'sequence_one_hot': torch.tensor(seq_one_hot, dtype=torch.float32),
            'contact_map': torch.tensor(contact_map, dtype=torch.float32),
        }
        
        if embeddings is not None:
            result['embeddings'] = torch.tensor(embeddings, dtype=torch.float32)
            
        return result
    
    def _sequence_to_one_hot(self, sequence):
        """Convert RNA sequence to one-hot encoding."""
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
        seq_len = len(sequence)
        one_hot = np.zeros((seq_len, 5), dtype=np.float32)
        
        for i, nt in enumerate(sequence):
            nt = nt.upper()
            if nt in mapping:
                one_hot[i, mapping[nt]] = 1.0
            else:
                # Handle unknown nucleotides with 'N'
                one_hot[i, mapping['N']] = 1.0
                
        return one_hot
    
    def _structure_to_contact_map(self, structure):
        """Convert dot-bracket notation to contact map."""
        seq_len = len(structure)
        contact_map = np.zeros((seq_len, seq_len), dtype=np.float32)
        
        # Parse dot-bracket notation to find base pairs
        stack = []
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    contact_map[i, j] = 1.0
                    contact_map[j, i] = 1.0
        
        return contact_map
    
    def _get_embeddings(self, sequence, seq_id=None):
        """Get RNA-FM embeddings for a sequence, with caching if possible."""
        # Check if embeddings are cached
        if self.embedding_cache_dir and seq_id:
            cache_path = Path(self.embedding_cache_dir) / f"{seq_id}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached embeddings: {e}")
        
        # Generate embeddings
        embeddings = self.embedding_model.get_embeddings(sequence, return_per_residue=True)
        
        # Cache embeddings if possible
        if self.embedding_cache_dir and seq_id:
            try:
                cache_path = Path(self.embedding_cache_dir) / f"{seq_id}.npy"
                np.save(cache_path, embeddings)
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")
        
        return embeddings

def get_rna_dataloader(data_path, split='train', batch_size=32, num_workers=4, 
                       max_seq_len=512, use_embeddings=True, embedding_cache_dir=None,
                       shuffle=True):
    """
    Create a DataLoader for the RNA dataset.
    
    Args:
        data_path (str): Path to dataset directory or file
        split (str): Data split ('train', 'val', or 'test')
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        max_seq_len (int): Maximum sequence length
        use_embeddings (bool): Whether to use RNA-FM embeddings
        embedding_cache_dir (str): Directory to cache embeddings
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    dataset = RNADataset(
        data_path=data_path,
        split=split,
        max_seq_len=max_seq_len,
        use_embeddings=use_embeddings,
        embedding_cache_dir=embedding_cache_dir
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_variable_length
    )

def collate_variable_length(batch):
    """
    Custom collate function to handle variable length sequences.
    
    Args:
        batch (list): List of dataset items
        
    Returns:
        dict: Collated batch with padded sequences
    """
    # Get maximum sequence length in the batch
    max_len = max(item['sequence_one_hot'].shape[0] for item in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    seq_one_hot = torch.zeros(batch_size, max_len, 5)
    contact_map = torch.zeros(batch_size, max_len, max_len)
    
    # If using embeddings, determine embedding dimension
    if 'embeddings' in batch[0]:
        embed_dim = batch[0]['embeddings'].shape[1]
        embeddings = torch.zeros(batch_size, max_len, embed_dim)
    else:
        embeddings = None
    
    # Collect metadata
    ids = []
    sequences = []
    structures = []
    seq_lengths = []
    
    # Fill in data
    for i, item in enumerate(batch):
        seq_len = item['sequence_one_hot'].shape[0]
        seq_lengths.append(seq_len)
        
        # Add sequence and structure info
        ids.append(item['id'])
        sequences.append(item['sequence'])
        structures.append(item['structure'])
        
        # Add one-hot sequence
        seq_one_hot[i, :seq_len, :] = item['sequence_one_hot']
        
        # Add contact map
        contact_map[i, :seq_len, :seq_len] = item['contact_map']
        
        # Add embeddings if present
        if embeddings is not None:
            embeddings[i, :seq_len, :] = item['embeddings']
    
    # Create result dictionary
    result = {
        'ids': ids,
        'sequences': sequences,
        'structures': structures,
        'sequence_one_hot': seq_one_hot,
        'contact_map': contact_map,
        'seq_lengths': torch.tensor(seq_lengths)
    }
    
    if embeddings is not None:
        result['embeddings'] = embeddings
        
    return result 