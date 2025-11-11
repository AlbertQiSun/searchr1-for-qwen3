#!/usr/bin/env python3
"""
Data preprocessing script for Search-R1 verl training.
Adds agent_name field and formats data for agentic RL training.
"""

import json
import os
import argparse
from typing import List, Dict, Any
import pandas as pd
from utils import load_dataset

def preprocess_search_r1_data(
    dataset_path: str,
    output_train_path: str,
    output_val_path: str,
    max_samples: int = None
):
    """
    Preprocess Search-R1 data for verl training.

    Args:
        dataset_path: Path to the raw dataset (JSONL)
        output_train_path: Path to save processed training data (Parquet)
        output_val_path: Path to save processed validation data (Parquet)
        max_samples: Maximum number of samples to process
    """

    print(f"Loading dataset from {dataset_path}")
    questions = load_dataset(dataset_path, max_samples)

    processed_data = []

    for question_data in questions:
        # Simple prompt format for verl
        prompt_text = f"Question: {question_data['question']}\n\nPlease reason step by step and provide a final answer."

        # Handle different data formats
        if "answer" in question_data:
            golden_answers = [question_data["answer"]]
        elif "golden_answers" in question_data:
            golden_answers = question_data["golden_answers"]
        else:
            golden_answers = ["Unknown"]

        # Create a sample with agent_name for verl
        sample = {
            "prompt": prompt_text,
            "response": "",  # Will be filled during rollout
            "reward": 0.0,   # Will be computed during training
            "agent_name": "search_agent_loop",  # Required for verl agentic RL
            "question_id": question_data.get("id", ""),
            "question": question_data["question"],
            "golden_answers": golden_answers
        }

        processed_data.append(sample)

    print(f"Processed {len(processed_data)} samples")

    # Split into train/val (80/20)
    train_size = int(0.8 * len(processed_data))
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Convert to DataFrames and save as Parquet
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Create output directories
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_path), exist_ok=True)

    # Save as Parquet format (required by verl)
    train_df.to_parquet(output_train_path, index=False)
    val_df.to_parquet(output_val_path, index=False)

    print(f"Training data saved to {output_train_path}")
    print(f"Validation data saved to {output_val_path}")

    # Print sample
    print("\nSample processed data:")
    print(json.dumps(train_data[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Search-R1 data for verl training")
    parser.add_argument("--dataset-path", default="datasets/hotpotqa/dev.jsonl",
                       help="Path to raw dataset")
    parser.add_argument("--output-train", default="data/search_r1_train.parquet",
                       help="Output path for training data")
    parser.add_argument("--output-val", default="data/search_r1_val.parquet",
                       help="Output path for validation data")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--tokenizer-path", default="/gpfsnyu/scratch/qs2196/.cache/models/Qwen3-1.7B",
                       help="Path to tokenizer model")

    args = parser.parse_args()

    preprocess_search_r1_data(
        args.dataset_path,
        args.output_train,
        args.output_val,
        args.max_samples
    )

