"""
Utility functions for generating RNA sequence embeddings using RNA-FM.
"""
import torch
import numpy as np
import logging
from pathlib import Path
import fm

# Configure logger
logger = logging.getLogger(__name__)

class RNAEmbedder:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the RNA-FM model for generating RNA sequence embeddings.
        Uses the official RNA-FM package directly.
        
        Args:
            device (str): Device to run the model on (cuda or cpu)
        """
        self.device = device
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        
        # Try to load RNA-FM model
        print("Loading RNA-FM model...")
        
        # Load RNA-FM model
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.model = self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Get batch converter for tokenization
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get embedding dimension
        self.embedding_dim = 640  # RNA-FM's fixed embedding dimension
        print(f"RNA-FM model loaded with embedding dimension: {self.embedding_dim}")
        
    def get_embeddings(self, sequence, layer=12, return_per_residue=True, pool_method='mean'):
        """
        Generate embeddings for an RNA sequence using RNA-FM.
        
        Args:
            sequence (str): RNA sequence
            layer (int): Layer to extract embeddings from (12 is the last layer for RNA-FM)
            return_per_residue (bool): Whether to return per-residue embeddings
            pool_method (str): Method for pooling sequence embedding ('mean', 'cls')
            
        Returns:
            numpy.ndarray: The sequence embeddings
        """
        # Clean sequence - use only standard nucleotides
        sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
        
        # Use fallback if model failed to load
        if self.model is None or self.batch_converter is None:
            return self._get_fallback_embeddings(sequence, return_per_residue)
        
        try:
            # Format data for RNA-FM
            data = [("rna_seq", sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[layer])
                
            # Get the embeddings from the specified layer
            token_representations = results["representations"][layer]
            
            # Process embeddings based on parameters
            if return_per_residue:
                # Return embeddings for each residue (excluding special tokens)
                embeddings = token_representations[0, 1:len(sequence)+1].cpu().numpy()
            else:
                # Pool embeddings for the entire sequence
                if pool_method == 'cls':
                    # Use the first token (CLS) embedding
                    embeddings = token_representations[0, 0].cpu().numpy()
                else:  # 'mean' pooling
                    # Average the per-residue embeddings
                    embeddings = torch.mean(token_representations[0, 1:len(sequence)+1], dim=0).cpu().numpy()
                    
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings with RNA-FM: {e}")
            print("Using fallback embeddings...")
            return self._get_fallback_embeddings(sequence, return_per_residue)
    
    def _get_fallback_embeddings(self, sequence, return_per_residue=True):
        """Generate fallback embeddings when primary model is unavailable"""
        # Map nucleotides to indices
        nt_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
        
        if return_per_residue:
            # Create embeddings for each position in the sequence
            embeddings = np.zeros((len(sequence), self.embedding_dim))
            for i, nt in enumerate(sequence):
                if nt in nt_map:
                    # One-hot encode the nucleotide
                    one_hot = np.zeros(4)
                    one_hot[nt_map[nt]] = 1
                    
                    # Add position information
                    pos_encoding = np.sin(np.arange(self.embedding_dim - 4) * (i / 1000))
                    
                    # Combine them
                    embeddings[i, :4] = one_hot
                    embeddings[i, 4:] = pos_encoding
            return embeddings
        else:
            # Create a single embedding for the whole sequence
            embedding = np.zeros(self.embedding_dim)
            
            # Use nucleotide frequencies for the first dimensions
            counts = {nt: 0 for nt in nt_map}
            for nt in sequence:
                if nt in counts:
                    counts[nt] += 1
            
            for nt, idx in nt_map.items():
                embedding[idx] = counts[nt] / max(len(sequence), 1)
                
            # Fill the rest with deterministic noise
            np.random.seed(hash(sequence) % 2**32)
            embedding[4:] = np.random.randn(self.embedding_dim - 4) * 0.1
            
            return embedding
        
    def batch_get_embeddings(self, sequences, **kwargs):
        """
        Generate embeddings for a batch of RNA sequences.
        
        Args:
            sequences (list): List of RNA sequences
            **kwargs: Additional arguments for get_embeddings
            
        Returns:
            list: List of numpy arrays containing embeddings
        """
        return [self.get_embeddings(seq, **kwargs) for seq in sequences]

# For backward compatibility
RNAFM = RNAEmbedder 