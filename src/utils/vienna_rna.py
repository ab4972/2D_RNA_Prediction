"""
Utility functions for RNA thermodynamic analysis using ViennaRNA.
"""
import RNA
import numpy as np
from scipy.sparse import csr_matrix
import os
import tempfile
import subprocess
import re

def get_basepair_probabilities(sequence, return_sparse=False):
    """
    Calculate base-pair probabilities using McCaskill's partition function algorithm via ViennaRNA.
    
    Args:
        sequence (str): RNA sequence
        return_sparse (bool): Whether to return sparse matrix
        
    Returns:
        numpy.ndarray: Base-pair probability matrix
    """
    # Clean sequence
    sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
    seq_length = len(sequence)
    
    # Initialize the probability matrix
    bp_matrix = np.zeros((seq_length, seq_length))

    # Use direct approach with Vienna RNA
    try:
        # Method 1: Use the low-level C function via RNA module for reliable access
        print("Using pf_fold for base pair probability calculation")
        structure, energy = RNA.pf_fold(sequence)
        print(f"Partition function MFE: {energy}")
        
        # Now extract the base pair probabilities from RNA's static variables
        for i in range(1, seq_length + 1):
            for j in range(i + 1, seq_length + 1):
                # Access the base pair probability table through the provided function
                prob = RNA.get_pr(i, j)
                if prob > 0:
                    # Convert to 0-based indexing for our matrix
                    bp_matrix[i-1, j-1] = prob
                    bp_matrix[j-1, i-1] = prob  # Make symmetric
    except Exception as e:
        print(f"Error using pf_fold: {str(e)}")
        
        try:
            # Method 2: Alternative method for newer ViennaRNA versions
            print("Trying alternate method with fold_compound")
            md = RNA.md()
            fc = RNA.fold_compound(sequence, md)
            fc.pf()
            
            # Try using pair_probs directly if available
            try:
                probs = fc.probs()  # May be available in some versions
                if isinstance(probs, list):
                    for pair in probs:
                        if len(pair) == 3:  # (i, j, probability) format
                            i, j, p = pair
                            if i > 0 and j > 0 and i <= seq_length and j <= seq_length:
                                bp_matrix[i-1, j-1] = p
                                bp_matrix[j-1, i-1] = p
                elif isinstance(probs, dict):
                    for (i, j), p in probs.items():
                        if i > 0 and j > 0 and i <= seq_length and j <= seq_length:
                            bp_matrix[i-1, j-1] = p
                            bp_matrix[j-1, i-1] = p
            except Exception as inner_e:
                print(f"Error accessing probabilities: {str(inner_e)}")
                
        except Exception as e2:
            print(f"Error using fold_compound: {str(e2)}")
            
    # Print statistics
    # print(f"ViennaRNA BPP Stats - Sum: {np.sum(bp_matrix):.2f}, Max: {np.max(bp_matrix):.4f}")
    # print(f"ViennaRNA BPP Stats - Non-zero entries: {np.count_nonzero(bp_matrix)}")
    
    if return_sparse:
        return csr_matrix(bp_matrix)
    
    return bp_matrix

def get_mfe_structure(sequence):
    """
    Get the minimum free energy (MFE) structure and energy.
    
    Args:
        sequence (str): RNA sequence
        
    Returns:
        tuple: (dot-bracket structure, free energy)
    """
    sequence = ''.join([c for c in sequence.upper() if c in 'AUCG'])
    (ss, mfe) = RNA.fold(sequence)
    return ss, mfe

def dotbracket_to_matrix(dotbracket):
    """
    Convert dot-bracket notation to contact matrix.
    
    Args:
        dotbracket (str): RNA structure in dot-bracket notation (supports extended notation with pseudoknots)
        
    Returns:
        numpy.ndarray: Binary contact matrix
    """
    seq_length = len(dotbracket)
    matrix = np.zeros((seq_length, seq_length))
    
    # Define all bracket types for extended dot-bracket notation
    opening_brackets = {'(': ')', '[': ']', '{': '}', '<': '>', 'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}
    closing_to_opening = {v: k for k, v in opening_brackets.items()}
    
    # Separate stack for each bracket type
    stacks = {bracket: [] for bracket in opening_brackets}
    
    # Parse dot-bracket
    for i, char in enumerate(dotbracket):
        if char in opening_brackets:
            stacks[char].append(i)
        elif char in closing_to_opening:
            opening = closing_to_opening[char]
            if stacks[opening]:  # Check if stack has elements
                j = stacks[opening].pop()
                matrix[j, i] = 1
                matrix[i, j] = 1
    
    return matrix

def matrix_to_dotbracket(matrix, threshold=0.5):
    """
    Convert a base-pair probability matrix to dot-bracket notation.
    
    Args:
        matrix (numpy.ndarray): Base-pair probability matrix
        threshold (float): Probability threshold for a base pair
        
    Returns:
        str: Dot-bracket notation
    """
    seq_length = matrix.shape[0]
    
    # Extract base pairs above threshold
    pairs = []
    for i in range(seq_length):
        for j in range(i+1, seq_length):
            if matrix[i, j] > threshold:
                pairs.append((i, j))
    
    # Sort by i position
    pairs.sort()
    
    # Assign brackets, handling pseudoknots
    structure = ['.' for _ in range(seq_length)]
    assigned = set()
    
    # Assign valid base pairs
    for i, j in pairs:
        if i not in assigned and j not in assigned:
            structure[i] = '('
            structure[j] = ')'
            assigned.add(i)
            assigned.add(j)
    
    return ''.join(structure)

def structure_distance(true_structure, pred_structure):
    """
    Calculate the base-pair distance between two structures.
    
    Args:
        true_structure (str): True structure in dot-bracket notation
        pred_structure (str): Predicted structure in dot-bracket notation
        
    Returns:
        int: Base-pair distance
    """
    return RNA.bp_distance(true_structure, pred_structure) 