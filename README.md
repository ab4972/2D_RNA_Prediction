# RNA Secondary Structure Prediction with GCNFold

This project implements an improved version of GCNFold for RNA secondary structure prediction, using RNA-FM embeddings and ViennaRNA base pair probabilities.

## Setup

1. Create a virtual environment for HuggingFace Token to load bprna dataset:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Before training the model, you need to download and prepare the dataset. The project includes two scripts for this purpose:

### 1. Downloading the Dataset

The `download_dataset.py` script downloads the bpRNA dataset from HuggingFace and saves it as CSV files.

#### Usage:
```bash
python src/data/download_dataset.py [options]
```

#### Options:
- `--dataset_name`: Name of the dataset on HuggingFace (default: "multimolecule/bprna")
- `--output_dir`: Directory to save the dataset (default: "data/bprna")

#### Example:
```bash
python src/data/download_dataset.py --output_dir "data/custom_path"
```

### 2. Cleaning the Dataset

The `clean_bprna_dataset.py` script processes the downloaded dataset to prepare it for training.

#### Usage:
```bash
python src/data/clean_bprna_dataset.py [options]
```

#### Options:
- `--input_file`: Path to the input CSV file (default: "data/bprna/train.csv")
- `--output_file`: Path to save the cleaned dataset (default: "data/bprna/train_clean.csv")
- `--min_length`: Minimum sequence length to keep (default: 20)
- `--max_length`: Maximum sequence length to keep (default: 150)

#### Example:
```bash
python src/data/clean_bprna_dataset.py --min_length 30 --max_length 200
```

## Training the Model

The `improved_gcnfold_training.py` script trains the GCNFold model on the bpRNA dataset.

### Usage:
```bash
python improved_gcnfold_training.py [options]
```

### Options:
- `--epochs`: Number of epochs to train (default: 20)
- `--max_length`: Maximum sequence length (default: 150)
- `--min_length`: Minimum sequence length (default: 20)
- `--pos_weight`: Positive weight for loss function (default: 10.0)
- `--lr`: Initial learning rate (default: 0.0001)
- `--seed`: Random seed (default: 42)
- `--max_samples`: Maximum number of samples to use and then splits to training, validation, and test sets (default: None)
- `--patience`: Patience for early stopping (default: 3)

### Example:
```bash
python improved_gcnfold_training.py --epochs 30 --max_length 200 --lr 0.0005
```

### Output:
The script creates a timestamped directory in `output/` containing:
- Trained model (`gcnfold_model.pt`)
- Training and validation loss curves
- ROC-AUC curves
- Precision-recall curves
- Confusion matrices
- Evaluation metrics
- Hyperparameter logs

## Visualizing RNA Structures

The `visualize_rna_structures.py` script generates visualizations of RNA structures and their predictions.

### Usage:
```bash
python visualize_rna_structures.py [options]
```

### Options:
- `--model_path`: Path to trained model (default: latest model in output/)
- `--data_path`: Path to RNA sequences (default: data/bprna/test_clean.csv)
- `--output_dir`: Directory to save visualizations (default: output/visualizations)
- `--num_samples`: Number of samples to visualize (default: 10)
- `--max_length`: Maximum sequence length (default: 150)

### Example:
```bash
python visualize_rna_structures.py --model_path output/gcnfold_improved_20250506_015201/gcnfold_model.pt
```

### Output:
The script generates visualizations for five seuqences of varying sizes in the specified visualization output directory:
- Structure heatmaps
- Base pair probability matrices
- Dot-bracket notation comparisons
- Performance metrics

## Model Architecture

The improved GCNFold model includes:
- RNA-FM embeddings (640-dimensional)
- 4 Graph Convolutional Network layers
- Hidden dimension of 256
- Minimum base pair distance of 3
- Structural constraints and stacking energy features
- ViennaRNA base pair probabilities as priors

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC

## Requirements

See `requirements.txt` for a complete list of dependencies. Key requirements include:
- PyTorch >= 2.0.0
- ViennaRNA >= 2.5.0
- RNA-FM embeddings
- DGL (Deep Graph Library)
- Other standard ML and scientific computing packages

## Notes

- The model uses curriculum learning for the first 3 epochs
- Early stopping is implemented with a patience of 3 epochs
- Base pair probabilities are normalized to log-odds form
- Structural constraints enforce RNA base pairing rules
- Stacking energies are applied to enhance prediction stability 