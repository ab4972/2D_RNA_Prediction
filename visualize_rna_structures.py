#!/usr/bin/env python3
"""
Visualize RNA secondary structure predictions for 5 sequences of varying sizes.
Compare GCNFold predictions, ViennaRNA predictions, and ground truth structures.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import RNA

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from src.models.gcnfold_improved import GCNFoldImproved
from src.utils.vienna_rna import dotbracket_to_matrix, matrix_to_dotbracket, get_basepair_probabilities
from src.utils.rna_embeddings import RNAFM
from improved_gcnfold_training import normalize_bp_priors

def load_sample_sequences(csv_path, size_ranges):
    """
    Load sample sequences from the dataset with different length ranges.
    
    Args:
        csv_path: Path to the dataset CSV file
        size_ranges: List of (min_len, max_len) tuples defining size ranges
        
    Returns:
        List of (seq_id, sequence, structure) tuples
    """
    print(f"Loading sample sequences from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    samples = []
    
    for min_len, max_len in size_ranges:
        # Filter by length range
        filtered = df[(df['sequence'].str.len() >= min_len) & (df['sequence'].str.len() <= max_len)]
        
        # Make sure sequences only contain standard nucleotides
        filtered = filtered[filtered['sequence'].apply(lambda x: all(c in 'AUCG' for c in x.upper()))]
        
        if not filtered.empty:
            # Select a random sample
            sample = filtered.sample(1).iloc[0]
            samples.append((sample['id'], sample['sequence'], sample['secondary_structure']))
            print(f"Selected {sample['id']} (length: {len(sample['sequence'])}) from range {min_len}-{max_len}")
    
    return samples

def get_vienna_rna_prediction(sequence):
    """
    Get secondary structure prediction from ViennaRNA.
    
    Args:
        sequence: RNA sequence
        
    Returns:
        Predicted structure in dot-bracket notation
    """
    # Use ViennaRNA to predict the structure
    fc = RNA.fold_compound(sequence)
    (structure, mfe) = fc.mfe()
    
    return structure, mfe

def visualize_structures(model, rna_fm, sample_data, device, output_dir, threshold=0.5):
    """
    Visualize RNA structures for sample sequences.
    
    Args:
        model: Trained GCNFold model
        rna_fm: RNA-FM embeddings model
        sample_data: List of (seq_id, sequence, structure) tuples
        device: PyTorch device
        output_dir: Directory to save visualizations
        threshold: Threshold for binary prediction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for seq_id, sequence, true_structure in sample_data:
            print(f"Processing {seq_id} (length: {len(sequence)})")
            
            # Get RNA-FM embeddings
            embeddings = rna_fm.get_embeddings(sequence)
            
            # Get ViennaRNA base-pair probabilities
            bp_probs = get_basepair_probabilities(sequence)
            normalized_bp_probs = normalize_bp_priors(bp_probs)
            
            # Get ViennaRNA structure prediction
            vienna_structure, mfe = get_vienna_rna_prediction(sequence)
            vienna_matrix = dotbracket_to_matrix(vienna_structure)
            
            # Convert ground truth structure to contact matrix
            true_matrix = dotbracket_to_matrix(true_structure)
            
            # Prepare inputs for model
            seq_len = len(sequence)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
            bp_priors_tensor = torch.tensor(normalized_bp_probs, dtype=torch.float32).unsqueeze(0).to(device)
            lengths_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)
            
            # Forward pass through model
            logits = model(embeddings_tensor, bp_priors_tensor, lengths_tensor)
            
            # Apply sigmoid and get predicted probabilities
            pred_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            # Create a mask for valid positions
            valid_mask = np.zeros_like(pred_probs, dtype=bool)
            for x in range(seq_len):
                for y in range(seq_len):
                    if x != y and abs(x - y) >= model.min_bp_distance:
                        valid_mask[x, y] = True
            
            # Apply mask to the probabilities
            masked_probs = np.zeros_like(pred_probs)
            masked_probs[valid_mask] = pred_probs[valid_mask]
            
            # Convert to binary predictions
            binary_preds = (masked_probs > threshold).astype(np.float32)
            
            # Convert binary predictions to dot-bracket notation
            pred_structure = matrix_to_dotbracket(binary_preds)
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            
            # Plot the RNA sequence
            axs[0, 0].text(0.5, 0.5, sequence, ha='center', va='center', fontsize=8, wrap=True)
            axs[0, 0].set_title(f'RNA Sequence: {seq_id}\nLength: {len(sequence)} nt')
            axs[0, 0].axis('off')
            
            # Plot the true contact map
            im1 = axs[0, 1].imshow(true_matrix, cmap='Blues', interpolation='none')
            axs[0, 1].set_title(f'Ground Truth')
            axs[0, 1].set_xlabel('Nucleotide Position')
            axs[0, 1].set_ylabel('Nucleotide Position')
            plt.colorbar(im1, ax=axs[0, 1])
            
            # Plot the ViennaRNA predicted contact map
            im2 = axs[0, 2].imshow(vienna_matrix, cmap='Greens', interpolation='none')
            axs[0, 2].set_title(f'ViennaRNA Prediction (MFE: {mfe:.2f})')
            axs[0, 2].set_xlabel('Nucleotide Position')
            axs[0, 2].set_ylabel('Nucleotide Position')
            plt.colorbar(im2, ax=axs[0, 2])
            
            # Plot the ViennaRNA base-pair probabilities
            im3 = axs[1, 0].imshow(bp_probs, cmap='viridis', interpolation='none')
            axs[1, 0].set_title('ViennaRNA Base-Pair Probabilities')
            axs[1, 0].set_xlabel('Nucleotide Position')
            axs[1, 0].set_ylabel('Nucleotide Position')
            plt.colorbar(im3, ax=axs[1, 0])
            
            # Plot the predicted probabilities
            im4 = axs[1, 1].imshow(masked_probs, cmap='hot', interpolation='none')
            axs[1, 1].set_title('GCNFold Predicted Probabilities')
            axs[1, 1].set_xlabel('Nucleotide Position')
            axs[1, 1].set_ylabel('Nucleotide Position')
            plt.colorbar(im4, ax=axs[1, 1])
            
            # Plot binary predictions
            im5 = axs[1, 2].imshow(binary_preds, cmap='Reds', interpolation='none')
            axs[1, 2].set_title(f'GCNFold Binary Predictions')
            axs[1, 2].set_xlabel('Nucleotide Position')
            axs[1, 2].set_ylabel('Nucleotide Position')
            plt.colorbar(im5, ax=axs[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{seq_id}_{len(sequence)}.png'), dpi=300)
            plt.close()
            
            print(f"  Saved visualization to {output_dir}/comparison_{seq_id}_{len(sequence)}.png")
            
            # Save structure comparison to text file
            with open(os.path.join(output_dir, f'comparison_{seq_id}_{len(sequence)}.txt'), 'w') as f:
                f.write(f"RNA Sequence ID: {seq_id}\n")
                f.write(f"Length: {len(sequence)} nt\n\n")
                f.write(f"Sequence: {sequence}\n\n")
                f.write(f"Ground Truth:   {true_structure}\n")
                f.write(f"ViennaRNA:      {vienna_structure}\n")
                f.write(f"GCNFold:        {pred_structure}\n")

def main():
    """Main function to visualize RNA structure predictions."""
    parser = argparse.ArgumentParser(description='Visualize RNA structure predictions for sample sequences')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/visualizations', help='Directory to save visualizations')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction (default: 0.5)')
    parser.add_argument('--csv_path', type=str, default='data/bprna/train_clean.csv', help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define size ranges for the 5 samples
    size_ranges = [
        (20, 40),    # Small
        (41, 60),    # Medium-small
        (61, 80),    # Medium
        (81, 100),   # Medium-large
        (101, 150)   # Large
    ]
    
    # Load sample sequences
    sample_data = load_sample_sequences(args.csv_path, size_ranges)
    
    # Initialize RNA-FM model
    print("Loading RNA-FM model...")
    rna_fm = RNAFM(device=device)
    print("RNA-FM model loaded!")
    
    # Initialize GCNFold model
    print("Loading GCNFold model...")
    model = GCNFoldImproved(
        embedding_dim=640,
        hidden_dim=256,
        num_gcn_layers=4,
        min_bp_distance=3,
        use_prior=True,
        apply_structural_constraints=True,
        apply_stacking_energy=True
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    # Generate visualizations
    visualize_structures(
        model=model,
        rna_fm=rna_fm,
        sample_data=sample_data,
        device=device,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 