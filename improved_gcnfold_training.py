"""
Improved training script for GCNFold with optimized parameters and architecture.
This version includes learning rate scheduling, reduced model complexity, and improved training stability.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import random
import time
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from src.models.gcnfold_improved import GCNFoldImproved
from src.utils.vienna_rna import dotbracket_to_matrix, get_basepair_probabilities
from src.utils.rna_embeddings import RNAFM

def load_bprna_data(csv_path, max_length=200, min_length=20, max_samples=None):
    """
    Load RNA sequences and their secondary structures from bpRNA dataset.
    
    Args:
        csv_path: Path to the bpRNA CSV file
        max_length: Maximum sequence length to include
        min_length: Minimum sequence length to include
        max_samples: Maximum number of samples to load
        
    Returns:
        list of (sequence_id, sequence, structure) tuples
    """
    print(f"Loading bpRNA data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter sequences by length
    filtered_data = []
    for _, row in df.iterrows():
        seq_id = row['id']
        sequence = row['sequence']
        structure = row['secondary_structure']
        
        # Check sequence length
        if min_length <= len(sequence) <= max_length:
            filtered_data.append((seq_id, sequence, structure))
    
    print(f"Found {len(filtered_data)} sequences with length between {min_length} and {max_length}")
    
    # Randomly sample if there are too many sequences
    if max_samples and len(filtered_data) > max_samples:
        filtered_data = random.sample(filtered_data, max_samples)
        print(f"Randomly sampled {max_samples} sequences")
    
    return filtered_data

def normalize_bp_priors(bp_probs):
    """
    Normalize base pair probabilities for better stability.
    
    Args:
        bp_probs: Base pair probability matrix
        
    Returns:
        Normalized base pair probability matrix
    """
    # Convert to log odds for better numeric stability
    epsilon = 1e-6  # Avoid log(0)
    bp_probs_clamped = np.clip(bp_probs, epsilon, 1.0 - epsilon)
    log_odds = np.log(bp_probs_clamped / (1.0 - bp_probs_clamped))
    
    # Scale log odds to reasonable range
    scaled_log_odds = np.clip(log_odds, -5.0, 5.0)
    
    return scaled_log_odds

def train_one_epoch(model, rna_fm, train_data, device, pos_weight=10.0, lr=0.0001, max_length=200, 
                   optimizer=None, accumulation_steps=4, print_every=10, use_curriculum=False):
    """
    Train the GCNFold model for one epoch with improved stability.
    
    Args:
        model: GCNFoldImproved model
        rna_fm: RNA-FM embeddings model
        train_data: List of (seq_id, sequence, structure) tuples for training
        device: PyTorch device
        pos_weight: Weight for positive examples in loss function
        lr: Learning rate
        max_length: Maximum sequence length
        optimizer: Optional optimizer to use (will create new one if None)
        accumulation_steps: Number of steps to accumulate gradients over
        print_every: How often to print batch statistics
        use_curriculum: Whether to use curriculum learning (easier examples first)
        
    Returns:
        Average training loss
    """
    print(f"Training GCNFold model for 1 epoch with lr={lr}, pos_weight={pos_weight}...")
    
    # Set model to training mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Start timing
    start_time = time.time()
    
    # Training variables
    train_loss = 0.0
    num_batches = 0
    
    # Sort data by length for curriculum learning if requested
    if use_curriculum:
        train_data = sorted(train_data, key=lambda x: len(x[1]))
    
    # Zero the gradients at the beginning
    optimizer.zero_grad()
    
    for i, (seq_id, sequence, structure) in enumerate(train_data):
        try:
            # Skip sequences that are too long
            if len(sequence) > max_length:
                print(f"Skipping {seq_id}: sequence too long ({len(sequence)} > {max_length})")
                continue
                
            # Clean sequence to ensure only AUCG nucleotides
            cleaned_sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
            
            # If sequence was changed, check structure compatibility
            if len(cleaned_sequence) != len(sequence):
                if len(cleaned_sequence) < 0.9 * len(sequence) or len(structure) != len(cleaned_sequence):
                    continue
            
            # Get RNA-FM embeddings
            embeddings = rna_fm.get_embeddings(cleaned_sequence)
            
            # Get ViennaRNA base-pair probabilities and normalize them
            bp_probs = get_basepair_probabilities(cleaned_sequence)
            normalized_bp_probs = normalize_bp_priors(bp_probs)
            
            # Convert structure to contact matrix
            target_matrix = dotbracket_to_matrix(structure)
            
            # Prepare inputs for model
            seq_len = len(cleaned_sequence)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(target_matrix, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Use normalized base-pair priors
            bp_priors_tensor = torch.tensor(normalized_bp_probs, dtype=torch.float32).unsqueeze(0).to(device)
            
            lengths_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(embeddings_tensor, bp_priors_tensor, lengths_tensor)
            
            # Create a mask for valid positions
            valid_mask = torch.zeros_like(target_tensor, dtype=torch.bool)
            for x in range(seq_len):
                for y in range(seq_len):
                    if x != y and abs(x - y) >= model.min_bp_distance:
                        valid_mask[0, x, y] = True
            
            # Apply mask to outputs and targets
            masked_logits = logits * valid_mask
            masked_targets = target_tensor * valid_mask
            
            # Compute loss
            loss = criterion(masked_logits, masked_targets)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update statistics
            train_loss += loss.item() * accumulation_steps  # Rescale for reporting
            num_batches += 1
            
            # Optimize every accumulation_steps or at the end
            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_data)):
                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            # Print batch statistics
            if (i+1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {i+1}/{len(train_data)}, Loss: {loss.item() * accumulation_steps:.6f}, Time elapsed: {elapsed:.2f}s")
        
        except Exception as e:
            print(f"Error processing training sample {seq_id}: {e}")
            continue
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    
    # Calculate average training loss
    avg_train_loss = train_loss / max(1, num_batches)
    
    print(f"Epoch completed in {training_time:.2f} seconds")
    print(f"Average training loss: {avg_train_loss:.6f}")
    
    return avg_train_loss

def plot_roc_curve(y_true, y_pred, output_path):
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        y_true: Flattened ground truth labels
        y_pred: Flattened predicted probabilities
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred, output_path):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Flattened ground truth labels
        y_pred: Flattened predicted probabilities
        output_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    return pr_auc

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Flattened ground truth labels
        y_pred: Flattened binary predictions
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['No Base Pair', 'Base Pair'])
    plt.yticks([0.5, 1.5], ['No Base Pair', 'Base Pair'])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return cm

def train_multiple_epochs(model, rna_fm, train_data, val_data, device, num_epochs=10, 
                         initial_lr=0.0001, pos_weight=10.0, max_length=200, patience=3):
    """
    Train the GCNFold model for multiple epochs with learning rate scheduling.
    
    Args:
        model: GCNFoldImproved model
        rna_fm: RNA-FM embeddings model
        train_data: List of (seq_id, sequence, structure) tuples for training
        val_data: List of (seq_id, sequence, structure) tuples for validation
        device: PyTorch device
        num_epochs: Number of epochs to train for
        initial_lr: Initial learning rate
        pos_weight: Weight for positive examples in loss function
        max_length: Maximum sequence length
        patience: Number of epochs to wait for validation improvement
        
    Returns:
        List of average training losses for each epoch, best model state dict
    """
    print(f"Training GCNFold model for {num_epochs} epochs...")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )
    
    # Training variables
    epoch_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement = 0
    
    # Use curriculum learning for early epochs
    use_curriculum = True
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # After a few epochs, switch to random order
        if epoch >= 3:
            use_curriculum = False
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        avg_loss = train_one_epoch(
            model=model,
            rna_fm=rna_fm,
            train_data=train_data,
            device=device,
            pos_weight=pos_weight,
            lr=current_lr,
            max_length=max_length,
            optimizer=optimizer,
            use_curriculum=use_curriculum
        )
        
        epoch_losses.append(avg_loss)
        
        # Evaluate on validation set if available
        if val_data:
            val_metrics, _, _ = evaluate_model(
                model=model,
                rna_fm=rna_fm,
                data=val_data,
                device=device,
                max_length=max_length,
                dataset_name="Validation"
            )
            val_loss = val_metrics['loss']
            val_losses.append(val_loss)
            
            print(f"Validation loss: {val_loss:.6f}")
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                print(f"New best validation loss: {best_val_loss:.6f}")
                no_improvement = 0
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")
            
            # Early stopping
            if no_improvement >= patience * 2:
                print(f"No improvement for {no_improvement} epochs. Early stopping.")
                break
    
    # If we didn't evaluate on validation set, use the final model
    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"output/loss_plot_{timestamp}.png")
    plt.close()
    
    return epoch_losses, val_losses, best_model_state

def evaluate_model(model, rna_fm, data, device, max_length=200, threshold=0.5, dataset_name="Evaluation"):
    """
    Evaluate the GCNFold model on a dataset.
    
    Args:
        model: GCNFoldImproved model
        rna_fm: RNA-FM embeddings model
        data: List of (seq_id, sequence, structure) tuples
        device: PyTorch device
        max_length: Maximum sequence length
        threshold: Threshold for binary prediction
        dataset_name: Name of the dataset for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating model on {dataset_name} set ({len(data)} sequences)...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluation variables
    all_predictions = []
    all_targets = []
    losses = []
    
    # For binary metrics
    all_flat_preds = []
    all_flat_targets = []
    
    # For ROC and PR curve
    all_flat_pred_probs = []
    
    with torch.no_grad():
        for i, (seq_id, sequence, structure) in enumerate(data):
            try:
                # Skip sequences that are too long
                if len(sequence) > max_length:
                    print(f"Skipping {seq_id}: sequence too long ({len(sequence)} > {max_length})")
                    continue
                
                # Clean sequence to ensure only AUCG nucleotides
                cleaned_sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
                
                # If the sequence length changed after cleaning, adjust the structure accordingly
                if len(cleaned_sequence) != len(sequence):
                    print(f"Warning: {seq_id} contains non-standard nucleotides. Length before: {len(sequence)}, after: {len(cleaned_sequence)}")
                    # Skip this sample if sequence is significantly altered
                    if len(cleaned_sequence) < 0.9 * len(sequence):
                        print(f"Skipping {seq_id}: too many non-standard nucleotides")
                        continue
                    
                    # Ensure structure length matches cleaned sequence length
                    if len(structure) != len(cleaned_sequence):
                        print(f"Skipping {seq_id}: structure length mismatch after cleaning")
                        continue
                
                # Get RNA-FM embeddings
                embeddings = rna_fm.get_embeddings(cleaned_sequence)
                
                # Get ViennaRNA base-pair probabilities and normalize
                bp_probs = get_basepair_probabilities(cleaned_sequence)
                normalized_bp_probs = normalize_bp_priors(bp_probs)
                
                # Convert structure to contact matrix
                target_matrix = dotbracket_to_matrix(structure)
                
                # Prepare inputs for model
                seq_len = len(cleaned_sequence)
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
                target_tensor = torch.tensor(target_matrix, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Use normalized ViennaRNA base-pair priors
                bp_priors_tensor = torch.tensor(normalized_bp_probs, dtype=torch.float32).unsqueeze(0).to(device)
                
                lengths_tensor = torch.tensor([seq_len], dtype=torch.long).to(device)
                
                # Forward pass
                logits = model(embeddings_tensor, bp_priors_tensor, lengths_tensor)
                
                # Create a mask for valid positions
                valid_mask = torch.zeros_like(target_tensor, dtype=torch.bool)
                for x in range(seq_len):
                    for y in range(seq_len):
                        if x != y and abs(x - y) >= model.min_bp_distance:
                            valid_mask[0, x, y] = True
                
                # Apply mask to outputs and targets
                masked_logits = logits * valid_mask
                masked_targets = target_tensor * valid_mask
                
                # Compute loss
                criterion = nn.BCEWithLogitsLoss(reduction='mean')
                loss = criterion(masked_logits, masked_targets)
                losses.append(loss.item())
                
                # Apply sigmoid and get predicted probabilities
                pred_probs = torch.sigmoid(masked_logits).cpu().numpy()
                
                # Convert to binary predictions
                pred_binary = (pred_probs > threshold).astype(np.float32)
                
                # Store for metrics calculation
                all_predictions.append(pred_binary[0])
                all_targets.append(target_matrix)
                
                # Flatten predictions and targets for binary metrics
                flat_preds_binary = pred_binary[0][valid_mask[0].cpu().numpy()]
                flat_targets = target_matrix[valid_mask[0].cpu().numpy()]
                flat_pred_probs = pred_probs[0][valid_mask[0].cpu().numpy()]
                
                all_flat_preds.extend(flat_preds_binary)
                all_flat_targets.extend(flat_targets)
                all_flat_pred_probs.extend(flat_pred_probs)
                
                # Print progress
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(data)} sequences")
            
            except Exception as e:
                print(f"Error processing evaluation sample {seq_id}: {e}")
                continue
    
    # Calculate average loss
    avg_loss = np.mean(losses) if losses else float('nan')
    
    # Calculate binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_flat_targets, all_flat_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_flat_targets, all_flat_preds)
    
    # Print evaluation results
    print(f"{dataset_name} Results:")
    print(f"  Average Loss: {avg_loss:.6f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': len(data),
        'all_flat_targets': np.array(all_flat_targets),
        'all_flat_preds': np.array(all_flat_preds),
        'all_flat_pred_probs': np.array(all_flat_pred_probs)
    }
    
    return metrics, all_predictions, all_targets

def main():
    """Main function to train and evaluate GCNFold with improved parameters."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate GCNFold model with improved parameters')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum sequence length (default: 150)')
    parser.add_argument('--min_length', type=int, default=20, help='Minimum sequence length (default: 20)')
    parser.add_argument('--pos_weight', type=float, default=10.0, help='Positive weight for loss function (default: 10.0)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (default: None)')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping (default: 3)')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"gcnfold_improved_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load bpRNA data
    all_data = load_bprna_data("data/bprna/train_clean.csv", max_length=args.max_length, min_length=args.min_length, max_samples=args.max_samples)
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split data into train, validation, and test sets (70%, 15%, 15%)
    total_samples = len(all_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size+val_size]
    test_data = all_data[train_size+val_size:]
    
    print(f"Dataset split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize RNA-FM model
    print("Loading RNA-FM model...")
    rna_fm = RNAFM(device=device)
    print("RNA-FM model loaded!")
    
    # Initialize GCNFold model with improved architecture
    print("Initializing GCNFold model with improved architecture...")
    model = GCNFoldImproved(
        embedding_dim=640,  # RNA-FM embedding dimension
        hidden_dim=256,
        num_gcn_layers=4,  # Reduced from 8 to 4
        min_bp_distance=3,
        use_prior=True,  
        apply_structural_constraints=True,
        apply_stacking_energy=True
    ).to(device)
    print("GCNFold model initialized!")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Log hyperparameters
    with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
        f.write("GCNFold Improved Training Hyperparameters\n")
        f.write("=======================================\n\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Positive weight: {args.pos_weight}\n")
        f.write(f"Max sequence length: {args.max_length}\n")
        f.write(f"Min sequence length: {args.min_length}\n")
        f.write(f"GCN layers: 4\n")
        f.write(f"Hidden dimension: 256\n")
        f.write(f"Patience: {args.patience}\n")
        f.write(f"Total model parameters: {total_params:,}\n")
        f.write(f"Dataset: {len(all_data)} sequences\n")
        f.write(f"Training set: {len(train_data)} sequences\n")
        f.write(f"Validation set: {len(val_data)} sequences\n")
        f.write(f"Test set: {len(test_data)} sequences\n")
    
    # Train the model for specified number of epochs
    epoch_losses, val_losses, best_model_state = train_multiple_epochs(
        model=model,
        rna_fm=rna_fm,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=args.epochs,
        initial_lr=args.lr,
        pos_weight=args.pos_weight,
        max_length=args.max_length,
        patience=args.patience
    )
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Evaluate the model on training set
    train_metrics, train_preds, train_targets = evaluate_model(
        model=model,
        rna_fm=rna_fm,
        data=train_data[:100],  # Use a subset for faster evaluation
        device=device,
        max_length=args.max_length,
        dataset_name="Training"
    )
    
    # Evaluate the model on validation set
    val_metrics, val_preds, val_targets = evaluate_model(
        model=model,
        rna_fm=rna_fm,
        data=val_data,
        device=device,
        max_length=args.max_length,
        dataset_name="Validation"
    )
    
    # Evaluate the model on test set
    test_metrics, test_preds, test_targets = evaluate_model(
        model=model,
        rna_fm=rna_fm,
        data=test_data,
        device=device,
        max_length=args.max_length,
        dataset_name="Test"
    )
    
    # Generate visualization plots
    
    # Loss curves (already plotted during training, but save data for record)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_curve_path)
    plt.close()
    
    # ROC curves
    roc_auc_train = plot_roc_curve(
        train_metrics['all_flat_targets'], 
        train_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "roc_curve_train.png")
    )
    
    roc_auc_val = plot_roc_curve(
        val_metrics['all_flat_targets'], 
        val_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "roc_curve_val.png")
    )
    
    roc_auc_test = plot_roc_curve(
        test_metrics['all_flat_targets'], 
        test_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "roc_curve_test.png")
    )
    
    # Precision-Recall curves
    pr_auc_train = plot_precision_recall_curve(
        train_metrics['all_flat_targets'], 
        train_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "pr_curve_train.png")
    )
    
    pr_auc_val = plot_precision_recall_curve(
        val_metrics['all_flat_targets'], 
        val_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "pr_curve_val.png")
    )
    
    pr_auc_test = plot_precision_recall_curve(
        test_metrics['all_flat_targets'], 
        test_metrics['all_flat_pred_probs'],
        os.path.join(output_dir, "pr_curve_test.png")
    )
    
    # Confusion matrices
    cm_train = plot_confusion_matrix(
        train_metrics['all_flat_targets'], 
        train_metrics['all_flat_preds'],
        os.path.join(output_dir, "confusion_matrix_train.png")
    )
    
    cm_val = plot_confusion_matrix(
        val_metrics['all_flat_targets'], 
        val_metrics['all_flat_preds'],
        os.path.join(output_dir, "confusion_matrix_val.png")
    )
    
    cm_test = plot_confusion_matrix(
        test_metrics['all_flat_targets'], 
        test_metrics['all_flat_preds'],
        os.path.join(output_dir, "confusion_matrix_test.png")
    )
    
    # Save model
    model_path = os.path.join(output_dir, "gcnfold_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write("GCNFold Improved Evaluation Results\n")
        f.write("==================================\n\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("Training Results:\n")
        for epoch, loss in enumerate(epoch_losses):
            f.write(f"  Epoch {epoch+1} loss: {loss:.6f}\n")
        f.write("\n")
        
        f.write("Evaluation Metrics:\n")
        for dataset_name, metrics, roc_auc, pr_auc in [
            ("Training", train_metrics, roc_auc_train, pr_auc_train), 
            ("Validation", val_metrics, roc_auc_val, pr_auc_val), 
            ("Test", test_metrics, roc_auc_test, pr_auc_test)
        ]:
            f.write(f"  {dataset_name} Set:\n")
            f.write(f"    Loss: {metrics['loss']:.6f}\n")
            f.write(f"    Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"    ROC-AUC: {roc_auc:.4f}\n")
            f.write(f"    PR-AUC: {pr_auc:.4f}\n\n")
    
    print("Evaluation complete!")
    print(f"Results saved to {output_dir}")
    print(f"Visualization plots saved in {output_dir}")

if __name__ == "__main__":
    main() 