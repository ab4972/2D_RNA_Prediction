"""
GCNFold Improved - An implementation of the GCNFold model described in:
'RNA Secondary Structure Prediction By Learning Unrolled Algorithms' (Zhang et al. 2020)
but adapted to use RNA-FM embeddings instead of one-hot encodings.

This implementation closely follows the original architecture but replaces the input encoding 
with pretrained RNA language model embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class FeatureEncoder(nn.Module):
    """
    Feature encoder that processes RNA-FM embeddings into node features for the GCN.
    This replaces the original one-hot encoding + CNN approach from the paper.
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(FeatureEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Process RNA-FM embeddings (640D) into node features
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, embeddings, sequence_lengths=None):
        """
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            sequence_lengths: [batch_size] - Optional length of each sequence
            
        Returns:
            node_features: [batch_size, seq_len, hidden_dim]
        """
        # Apply linear transformation to embeddings
        node_features = self.transform(embeddings)
        
        return node_features

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer for learning RNA structure features.
    """
    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Transform for self-loop
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        
        # Transform for edges
        self.W_edge = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, node_features, adjacency_matrix=None):
        """
        Args:
            node_features: [batch_size, seq_len, hidden_dim]
            adjacency_matrix: [batch_size, seq_len, seq_len] or None (fully connected)
            
        Returns:
            updated_features: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = node_features.shape
        
        # Self-loop transformation
        self_features = self.W_self(node_features)
        
        # Edge transformations (aggregation from neighbors)
        # If adjacency matrix is None, assume fully connected
        if adjacency_matrix is None:
            # For each node, aggregate features from all other nodes
            neighbor_features = torch.matmul(
                torch.ones((batch_size, seq_len, seq_len), device=node_features.device), 
                self.W_edge(node_features)
            )
        else:
            # Use adjacency matrix to determine which nodes to aggregate from
            neighbor_features = torch.matmul(adjacency_matrix, self.W_edge(node_features))
        
        # Combine self and neighbor features with ReLU activation
        combined_features = self_features + neighbor_features
        updated_features = F.relu(combined_features)
        
        # Apply layer normalization
        updated_features = self.layer_norm(updated_features)
        
        return updated_features

class PairwiseScorer(nn.Module):
    """
    MLP that takes pairs of node features and computes a score for each potential base pair.
    """
    def __init__(self, hidden_dim, use_prior=True):
        super(PairwiseScorer, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_prior = use_prior
        
        # Input features will be 2*hidden_dim (concatenated node features) + 1 (prior prob)
        input_size = 2 * hidden_dim + 1 if use_prior else 2 * hidden_dim
        
        # MLP to score potential base pairs
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, node_features, bp_priors=None):
        """
        Args:
            node_features: [batch_size, seq_len, hidden_dim]
            bp_priors: [batch_size, seq_len, seq_len] - Prior base pair probabilities (optional)
            
        Returns:
            bp_scores: [batch_size, seq_len, seq_len] - Base pair scores
        """
        batch_size, seq_len, hidden_dim = node_features.shape
        
        # Expand node features for pairwise comparison
        node_i = node_features.unsqueeze(2).expand(batch_size, seq_len, seq_len, hidden_dim)
        node_j = node_features.unsqueeze(1).expand(batch_size, seq_len, seq_len, hidden_dim)
        
        # Concatenate features
        pair_features = torch.cat([node_i, node_j], dim=-1)  # [batch_size, seq_len, seq_len, 2*hidden_dim]
        
        # Include prior probabilities if provided
        if self.use_prior and bp_priors is not None:
            priors = bp_priors.unsqueeze(-1)  # [batch_size, seq_len, seq_len, 1]
            pair_features = torch.cat([pair_features, priors], dim=-1)
        
        # Reshape for MLP
        flat_features = pair_features.view(batch_size * seq_len * seq_len, -1)
        
        # Compute scores
        flat_scores = self.mlp(flat_features)
        
        # Reshape back to pairwise matrix
        bp_scores = flat_scores.view(batch_size, seq_len, seq_len)
        
        return bp_scores

class StructuralConstraints(nn.Module):
    """
    Module for applying RNA structural constraints.
    This implements various biophysical constraints for RNA secondary structure.
    """
    def __init__(self, min_bp_distance=3, apply_stacking_energy=True):
        super(StructuralConstraints, self).__init__()
        self.min_bp_distance = min_bp_distance
        self.apply_stacking_energy = apply_stacking_energy
        
        # Define valid base pairs (Watson-Crick and Wobble pairs)
        # A-U, G-C, G-U are valid RNA base pairs
        self.valid_pairs = {
            'A': ['U'],
            'U': ['A', 'G'],
            'G': ['C', 'U'],
            'C': ['G']
        }
        
        # Stacking energy parameters (simplified version from Turner energy model)
        # Higher values indicate more stable stacking
        self.stacking_energies = {
            ('A', 'U', 'A', 'U'): 0.9,  # A-U stacked with A-U
            ('A', 'U', 'G', 'C'): 1.1,  # A-U stacked with G-C
            ('A', 'U', 'G', 'U'): 0.8,  # A-U stacked with G-U
            ('G', 'C', 'A', 'U'): 1.1,  # G-C stacked with A-U
            ('G', 'C', 'G', 'C'): 1.3,  # G-C stacked with G-C
            ('G', 'C', 'G', 'U'): 1.0,  # G-C stacked with G-U
            ('G', 'U', 'A', 'U'): 0.8,  # G-U stacked with A-U
            ('G', 'U', 'G', 'C'): 1.0,  # G-U stacked with G-C
            ('G', 'U', 'G', 'U'): 0.7,  # G-U stacked with G-U
        }
        
        # Default energy for unlisted pairs
        self.default_stacking_energy = 0.5
    
    def is_valid_base_pair(self, nt_i, nt_j):
        """Check if two nucleotides can form a valid base pair"""
        nt_i = nt_i.upper()
        nt_j = nt_j.upper()
        
        if nt_i in self.valid_pairs and nt_j in self.valid_pairs[nt_i]:
            return True
        return False
    
    def get_stacking_energy(self, nt_i, nt_j, nt_i_next, nt_j_prev):
        """Get stacking energy for adjacent base pairs (i,j) and (i+1,j-1)"""
        key = (nt_i, nt_j, nt_i_next, nt_j_prev)
        
        # Check for valid stacking (need valid base pairs)
        if (self.is_valid_base_pair(nt_i, nt_j) and 
            self.is_valid_base_pair(nt_i_next, nt_j_prev)):
            return self.stacking_energies.get(key, self.default_stacking_energy)
        return 0.0
    
    def create_base_pair_mask(self, sequences, sequence_lengths):
        """
        Create a mask for valid base pairs based on sequence.
        
        Args:
            sequences: list of RNA sequences (batch)
            sequence_lengths: [batch_size] tensor of sequence lengths
            
        Returns:
            mask: [batch_size, seq_len, seq_len] tensor (1 for valid pairs, 0 for invalid)
        """
        batch_size = len(sequences)
        max_len = max(sequence_lengths).item()
        device = sequence_lengths.device
        
        # Initialize mask
        mask = torch.zeros((batch_size, max_len, max_len), device=device)
        
        # Apply base-pairing rules
        for b, seq in enumerate(sequences):
            length = sequence_lengths[b].item()
            
            for i in range(length):
                for j in range(i + self.min_bp_distance, length):
                    # Apply minimum distance constraint
                    if j - i < self.min_bp_distance:
                        continue
                    
                    # Apply base-pairing constraint (Watson-Crick and wobble pairs)
                    nt_i = seq[i].upper()
                    nt_j = seq[j].upper()
                    
                    if nt_j in self.valid_pairs.get(nt_i, []):
                        mask[b, i, j] = 1.0
                        mask[b, j, i] = 1.0  # Symmetrical
        
        return mask
    
    def apply_stacking_energies(self, bp_scores, sequences, sequence_lengths):
        """
        Apply stacking energy bonuses to base pair scores.
        
        Args:
            bp_scores: [batch_size, seq_len, seq_len] - Base pair scores
            sequences: list of RNA sequences (batch)
            sequence_lengths: [batch_size] tensor of sequence lengths
            
        Returns:
            updated_scores: [batch_size, seq_len, seq_len] - Updated scores with stacking energies
        """
        if not self.apply_stacking_energy:
            return bp_scores
            
        batch_size = len(sequences)
        device = bp_scores.device
        
        # Create a copy to avoid modifying the original
        updated_scores = bp_scores.clone()
        
        # Apply stacking energies
        for b, seq in enumerate(sequences):
            length = sequence_lengths[b].item()
            
            for i in range(length - 1):  # Exclude last position
                for j in range(i + self.min_bp_distance + 1, length):  # Must have room for i+1,j-1 pair
                    if j <= i + self.min_bp_distance or j <= 1:
                        continue
                        
                    # Check if (i,j) and (i+1,j-1) can form stacked base pairs
                    nt_i = seq[i]
                    nt_j = seq[j]
                    nt_i_next = seq[i+1]
                    nt_j_prev = seq[j-1]
                    
                    stacking_energy = self.get_stacking_energy(nt_i, nt_j, nt_i_next, nt_j_prev)
                    
                    if stacking_energy > 0:
                        # Apply stacking energy bonus
                        updated_scores[b, i, j] += stacking_energy
                        updated_scores[b, j, i] += stacking_energy  # Symmetrical
                        
                        # Also boost the complementary base pair
                        updated_scores[b, i+1, j-1] += stacking_energy
                        updated_scores[b, j-1, i+1] += stacking_energy  # Symmetrical
        
        return updated_scores

class GCNFoldImproved(nn.Module):
    """
    Improved GCNFold model that uses RNA-FM embeddings and follows the original paper's architecture.
    """
    def __init__(self, embedding_dim=640, hidden_dim=256, num_gcn_layers=3, min_bp_distance=3, use_prior=True, 
                 apply_structural_constraints=True, apply_stacking_energy=True):
        super(GCNFoldImproved, self).__init__()
        self.embedding_dim = embedding_dim  # RNA-FM embedding dimension (640)
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.min_bp_distance = min_bp_distance
        self.use_prior = use_prior
        self.apply_structural_constraints = apply_structural_constraints
        
        # Feature encoder (RNA-FM embeddings → node features)
        self.feature_encoder = FeatureEncoder(embedding_dim, hidden_dim)
        
        # GCN layers for message passing
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Base pair scorer
        self.pair_scorer = PairwiseScorer(hidden_dim, use_prior=use_prior)
        
        # Structural constraints handler
        if apply_structural_constraints:
            self.structural_constraints = StructuralConstraints(
                min_bp_distance=min_bp_distance,
                apply_stacking_energy=apply_stacking_energy
            )
    
    def forward(self, embeddings, bp_priors, sequence_lengths, sequences=None):
        """
        Forward pass through GCNFold.
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim] - RNA-FM embeddings
            bp_priors: [batch_size, seq_len, seq_len] - Prior base pair probabilities from ViennaRNA
            sequence_lengths: [batch_size] - Length of each sequence
            sequences: list of RNA sequences (needed for structural constraints)
            
        Returns:
            bp_scores: [batch_size, seq_len, seq_len] - Logit scores for base pairs (sigmoid to get probs)
        """
        # Encode input embeddings into node features
        node_features = self.feature_encoder(embeddings, sequence_lengths)
        
        # Pass through GCN layers
        for gcn_layer in self.gcn_layers:
            node_features = gcn_layer(node_features)
        
        # Compute base pair scores
        bp_scores = self.pair_scorer(node_features, bp_priors if self.use_prior else None)
        
        # Apply constraints (e.g., no base pairs within min_bp_distance)
        batch_size, seq_len, _ = bp_scores.shape
        mask = torch.ones_like(bp_scores)
        
        for b in range(batch_size):
            length = sequence_lengths[b].item()
            
            # Apply minimum base pair distance constraint
            for i in range(length):
                for j in range(length):
                    if abs(i - j) < self.min_bp_distance:
                        mask[b, i, j] = 0.0
            
            # Zero-out padding positions
            if length < seq_len:  # Only apply if there is padding
                mask[b, length:, :] = 0.0
                mask[b, :, length:] = 0.0
        
        # Apply basic constraints mask
        bp_scores = bp_scores * mask
        
        # Apply structural constraints if sequences are provided
        if self.apply_structural_constraints and sequences is not None:
            # Apply base-pairing constraints (Watson-Crick and wobble pairs)
            base_pair_mask = self.structural_constraints.create_base_pair_mask(sequences, sequence_lengths)
            bp_scores = bp_scores * base_pair_mask
            
            # Apply stacking energies to enhance scores for stable stacks
            bp_scores = self.structural_constraints.apply_stacking_energies(bp_scores, sequences, sequence_lengths)
        
        return bp_scores
    
    def _enforce_pseudoknot_free(self, bp_probs, sequence_lengths, threshold=0.5):
        """
        Enforce pseudoknot-free constraint on base pair probabilities
        using a dynamic programming approach similar to the Nussinov algorithm.
        
        Args:
            bp_probs: [batch_size, seq_len, seq_len] - Base pair probabilities
            sequence_lengths: [batch_size] - Length of each sequence
            threshold: Probability threshold for including a base pair
            
        Returns:
            filtered_bp_probs: [batch_size, seq_len, seq_len] - Filtered probabilities
        """
        batch_size = bp_probs.shape[0]
        max_len = bp_probs.shape[1]
        device = bp_probs.device
        
        # Create a copy to avoid modifying the original
        filtered_probs = bp_probs.clone()
        
        for b in range(batch_size):
            length = sequence_lengths[b].item()
            
            # Convert probabilities to binary (above threshold)
            binary_map = (bp_probs[b, :length, :length] > threshold).float()
            
            # Find maximum weight non-crossing matching using dynamic programming
            # This is a simplification of the maximum weight non-crossing matching algorithm
            
            # Initialize DP table
            dp = torch.zeros((length, length), device=device)
            bp_used = torch.zeros((length, length), device=device, dtype=torch.bool)
            
            # Fill DP table (variant of Nussinov algorithm)
            for span in range(self.min_bp_distance + 1, length):
                for i in range(length - span):
                    j = i + span
                    
                    # Option 1: j is unpaired
                    dp[i, j] = dp[i, j-1]
                    
                    # Option 2: j pairs with some k in [i, j-1]
                    for k in range(i, j - self.min_bp_distance + 1):
                        if binary_map[k, j] > 0:  # If potential base pair exists
                            new_score = dp[i, k-1] + dp[k+1, j-1] + bp_probs[b, k, j].item()
                            if new_score > dp[i, j]:
                                dp[i, j] = new_score
                                bp_used[k, j] = True
            
            # Backtrack to find valid base pairs
            valid_bp = torch.zeros((length, length), device=device)
            stack = [(0, length - 1)]
            
            while stack:
                i, j = stack.pop()
                if i >= j:
                    continue
                    
                found_pair = False
                for k in range(i, j - self.min_bp_distance + 1):
                    if bp_used[k, j]:
                        valid_bp[k, j] = 1
                        valid_bp[j, k] = 1  # Symmetric
                        
                        # Push regions to stack
                        if k > i:
                            stack.append((i, k-1))
                        if k+1 < j:
                            stack.append((k+1, j-1))
                        found_pair = True
                        break
                        
                if not found_pair and j > i:
                    stack.append((i, j-1))
            
            # Apply the valid base pair mask
            for i in range(length):
                for j in range(length):
                    if valid_bp[i, j] == 0:
                        filtered_probs[b, i, j] = 0.0
        
        return filtered_probs
    
    def predict_structure(self, bp_probs, sequence_lengths, sequences=None, threshold=0.5, enforce_pseudoknot_free=True):
        """
        Convert base pair probabilities to dot-bracket notation.
        
        Args:
            bp_probs: [batch_size, seq_len, seq_len] - Base pair probabilities
            sequence_lengths: [batch_size] - Length of each sequence
            sequences: list of RNA sequences (optional, for structural constraints)
            threshold: float - Threshold for determining base pairs
            enforce_pseudoknot_free: bool - Whether to enforce pseudoknot-free constraint
            
        Returns:
            structures: list of strings - Dot-bracket notation for each sequence
        """
        batch_size = bp_probs.shape[0]
        structures = []
        
        # Apply pseudoknot-free constraint if requested
        if enforce_pseudoknot_free:
            bp_probs = self._enforce_pseudoknot_free(bp_probs, sequence_lengths, threshold)
        
        # Apply structural constraints if sequences are provided
        if self.apply_structural_constraints and sequences is not None:
            # Create base pair mask from sequences
            base_pair_mask = self.structural_constraints.create_base_pair_mask(sequences, sequence_lengths)
            bp_probs = bp_probs * base_pair_mask
        
        # Process each sequence in the batch
        for b in range(batch_size):
            length = sequence_lengths[b].item()
            structure = ['.' for _ in range(length)]
            
            # Sort base pairs by probability
            pairs = []
            for i in range(length):
                for j in range(i+1, length):  # Only consider i < j
                    prob = bp_probs[b, i, j].item()
                    if prob > threshold:
                        pairs.append((i, j, prob))
            
            # Sort by probability (highest first)
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Assign base pairs, ensuring no conflicts
            paired = set()
            for i, j, _ in pairs:
                if i not in paired and j not in paired:
                    structure[i] = '('
                    structure[j] = ')'
                    paired.add(i)
                    paired.add(j)
            
            structures.append(''.join(structure))
        
        return structures 