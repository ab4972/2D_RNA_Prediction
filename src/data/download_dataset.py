"""
Download the multimolecule/bprna dataset from HuggingFace.
"""
import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd

def main(args):
    # Load environment variables (.env file)
    load_dotenv()
    
    # Get HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HuggingFace token not found. Make sure to set it in the .env file.")
    
    print(f"Downloading dataset: {args.dataset_name}")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset(args.dataset_name, use_auth_token=hf_token)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save each split to CSV
    for split, data in dataset.items():
        output_path = os.path.join(args.output_dir, f"{split}.csv")
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved {split} split to {output_path} ({len(df)} examples)")
    
    print("Dataset download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the bpRNA dataset from HuggingFace")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="multimolecule/bprna", 
        help="Name of the dataset on HuggingFace"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/bprna", 
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    main(args) 