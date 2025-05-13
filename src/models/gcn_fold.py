"""
GCNFold-inspired model for RNA structure prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import numpy as np


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, activation=F.relu):
        super(GCNLayer, self).__init__()
        self.conv = GraphConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def forward(self, g, feat):
        h = self.conv(g, feat)
        h = self.activation(h)
        h = self.dropout(h)
        return h


class MLP(nn.Module):
    """
    Multi-layer perceptron for feature transformation.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation/dropout for output layer
                x = F.relu(x)
                x = self.dropout(x)
        return x


class EdgePredictor(nn.Module):
    """
    Edge prediction module to output logits for base-pair prediction.
    """
    def __init__(self, node_dim):
        super(EdgePredictor, self).__init__()
        
        # Edge prediction network (combine node features)
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1)
            # Sigmoid is applied in forward() rather than here
        )
        
    def forward(self, node_features):
        batch_size, seq_len, feat_dim = node_features.shape
        
        # Create all possible pairs of nucleotide features
        node_i = node_features.unsqueeze(2).expand(batch_size, seq_len, seq_len, feat_dim)
        node_j = node_features.unsqueeze(1).expand(batch_size, seq_len, seq_len, feat_dim)
        
        # Concatenate the features of pairs
        pair_feats = torch.cat([node_i, node_j], dim=-1)
        
        # Predict edge scores (reshape for efficient processing)
        pair_feats_flat = pair_feats.view(-1, feat_dim * 2)
        edge_logits_flat = self.edge_predictor(pair_feats_flat)
        edge_logits = edge_logits_flat.view(batch_size, seq_len, seq_len)
        
        return edge_logits


class GCNFold(nn.Module):
    """
    GCNFold-inspired model for RNA structure prediction.
    """
    def __init__(
        self,
        embedding_dim=640,  # RNA-FM embedding dimension
        hidden_dim=256,
        num_gcn_layers=3,
        gcn_dropout=0.1,
        use_prior=True,  # Whether to use ViennaRNA prior
        min_bp_distance=3,  # Minimum base-pair distance
    ):
        super(GCNFold, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.use_prior = use_prior
        self.min_bp_distance = min_bp_distance
        
        # Feature transformation for RNA-FM embeddings
        self.embedding_transform = MLP(
            in_dim=embedding_dim,
            hidden_dim=hidden_dim*2,
            out_dim=hidden_dim,
            num_layers=2,
        )
        
        # Feature transformation for ViennaRNA prior (if used)
        if use_prior:
            self.prior_weight = nn.Parameter(torch.tensor(0.5))
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_feats = hidden_dim
            out_feats = hidden_dim
            self.gcn_layers.append(
                GCNLayer(in_feats, out_feats, dropout=gcn_dropout)
            )
            
        # Edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim)
    
    def _build_graph(self, seq_len, basepair_probs=None, threshold=0.01, device='cpu'):
        """
        Build a graph from sequence and optionally basepair probabilities.
        
        Args:
            seq_len (int): Sequence length
            basepair_probs (torch.Tensor): Base-pair probabilities (B, L, L)
            threshold (float): Threshold for base-pair probability
            device (str): Device to create graph on
            
        Returns:
            dgl.DGLGraph: Graph representation of RNA sequence
        """
        # Initialize graph with local connections
        src_local = []
        dst_local = []
        
        # Add local connections (sequential neighbors)
        for i in range(seq_len):
            # Connect to neighbors within window of 2
            for j in range(max(0, i-2), min(seq_len, i+3)):
                if i != j:  # No self-loops
                    src_local.append(i)
                    dst_local.append(j)
        
        # Create graph
        g = dgl.graph((torch.tensor(src_local, device=device), 
                      torch.tensor(dst_local, device=device)), 
                     num_nodes=seq_len)
        
        # If base-pair probabilities are provided, add long-range edges
        if basepair_probs is not None:
            # Get edges where bp probability > threshold and |i-j| > min_bp_distance
            bp_mask = torch.zeros_like(basepair_probs, dtype=torch.bool)
            for i in range(seq_len):
                for j in range(seq_len):
                    if (abs(i - j) >= self.min_bp_distance and 
                        basepair_probs[i, j] > threshold):
                        bp_mask[i, j] = True
            
            # Extract source and destination indices
            src_bp, dst_bp = torch.where(bp_mask)
            
            # Add to graph if there are any base-pair edges
            if len(src_bp) > 0:
                g.add_edges(src_bp, dst_bp)
        
        return g
    
    def _build_batch_graph(self, lengths, basepair_probs=None, threshold=0.01):
        """
        Build a batch of graphs.
        
        Args:
            lengths (torch.Tensor): Sequence lengths (B,)
            basepair_probs (torch.Tensor): Base-pair probabilities (B, L, L)
            threshold (float): Threshold for base-pair probability
            
        Returns:
            dgl.DGLGraph: Batched graph
        """
        batch_size = lengths.shape[0]
        device = lengths.device
        
        # Create a list of graphs
        graphs = []
        for i in range(batch_size):
            seq_len = lengths[i].item()
            bp_probs_i = None
            if basepair_probs is not None:
                bp_probs_i = basepair_probs[i, :seq_len, :seq_len]
            
            g = self._build_graph(seq_len, bp_probs_i, threshold, device)
            graphs.append(g)
        
        # Batch the graphs
        batched_graph = dgl.batch(graphs)
        return batched_graph
    
    def forward(self, embeddings, bp_probs, lengths):
        """
        Forward pass of the GCNFold model.
        
        Args:
            embeddings (torch.Tensor): RNA-FM embeddings (B, L, D)
            bp_probs (torch.Tensor): ViennaRNA base-pair probabilities (B, L, L)
            lengths (torch.Tensor): Sequence lengths (B,)
            
        Returns:
            torch.Tensor: Predicted base-pair probabilities (B, L, L)
        """
        batch_size = embeddings.shape[0]
        max_len = embeddings.shape[1]
        device = embeddings.device
        
        # Create mask for valid positions
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        
        # Transform RNA-FM embeddings
        node_feats = self.embedding_transform(embeddings)
        
        # Create a batch of graphs
        if self.use_prior:
            graphs = self._build_batch_graph(lengths, bp_probs)
        else:
            graphs = self._build_batch_graph(lengths)
        
        # Apply GCN layers
        h = node_feats.reshape(-1, self.hidden_dim)  # Flatten batch for DGL
        
        for i in range(self.num_gcn_layers):
            h = self.gcn_layers[i](graphs, h)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        node_embeddings = h.reshape(batch_size, max_len, self.hidden_dim)
        
        # Predict edge scores
        pred_bp_logits = self.edge_predictor(node_embeddings)
        
        # Apply mask for valid positions
        valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        
        # Apply minimum base-pair distance constraint
        dist_mask = torch.zeros((max_len, max_len), dtype=torch.bool, device=device)
        for i in range(max_len):
            for j in range(max_len):
                if abs(i - j) >= self.min_bp_distance:
                    dist_mask[i, j] = True
        
        combined_mask = valid_mask & dist_mask.unsqueeze(0)
        pred_bp_logits = pred_bp_logits * combined_mask
        
        # Combine with ViennaRNA prior if used
        if self.use_prior:
            prior_weight = torch.sigmoid(self.prior_weight)
            # Apply sigmoid to convert logits to probabilities
            pred_bp_probs = torch.sigmoid(pred_bp_logits)
            pred_bp_probs = (1 - prior_weight) * pred_bp_probs + prior_weight * bp_probs
            return pred_bp_probs
        
        # Apply sigmoid to convert logits to probabilities
        pred_bp_probs = torch.sigmoid(pred_bp_logits)
        return pred_bp_probs
    
    def predict_structure(self, pred_bp_probs, lengths, threshold=0.5):
        """
        Convert base-pair probabilities to dot-bracket notation.
        
        Args:
            pred_bp_probs (torch.Tensor): Predicted base-pair probabilities (B, L, L)
            lengths (torch.Tensor): Sequence lengths (B,)
            threshold (float): Probability threshold for base pairs
            
        Returns:
            list: List of dot-bracket structures
        """
        batch_size = pred_bp_probs.shape[0]
        structures = []
        
        for i in range(batch_size):
            seq_len = lengths[i].item()
            bp_probs = pred_bp_probs[i, :seq_len, :seq_len].cpu().numpy()
            
            # Extract the most likely base pairs
            structure = ['.' for _ in range(seq_len)]
            used = set()
            
            # Create a list of base pairs by probability
            pairs = []
            for x in range(seq_len):
                for y in range(x + self.min_bp_distance, seq_len):
                    pairs.append((x, y, bp_probs[x, y]))
            
            # Sort by probability (descending)
            pairs.sort(key=lambda p: p[2], reverse=True)
            
            # Assign base pairs
            for x, y, prob in pairs:
                if prob < threshold:  # Skip low-probability pairs
                    continue
                if x not in used and y not in used:
                    structure[x] = '('
                    structure[y] = ')'
                    used.add(x)
                    used.add(y)
            
            structures.append(''.join(structure))
        
        return structures 