#!/usr/bin/env python
"""
Script to download HuggingFace models for ASR tasks.
Usage:
    python download_model.py --model_name MODEL_NAME [--path PATH] [--revision REVISION]
"""

import argparse
import os
from huggingface_hub import snapshot_download


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download a model from HuggingFace Hub.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to download (e.g., facebook/wav2vec2-base-960h)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Path where the model will be downloaded. Default is 'asr/models/'."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Specific model revision to download. Default is 'main'."
    )
    parser.add_argument(
        "--local_dir_name",
        type=str,
        default=None,
        help="Optional local directory name for the model. If not provided, will use the model name."
    )

    return parser.parse_args()


def download_model(model_name, path, revision="main", local_dir_name=None):
    """
    Download a model from HuggingFace Hub.

    Args:
        model_name (str): Name of the model to download (e.g., facebook/wav2vec2-base-960h)
        path (str): Path where the model will be downloaded
        revision (str): Specific model revision to download
        local_dir_name (str, optional): Custom directory name for the downloaded model

    Returns:
        Path to the downloaded model
    """
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Set the local directory name
    if local_dir_name is None:
        # Use the last part of the model name as directory name
        # e.g., facebook/wav2vec2-base-960h -> wav2vec2-base-960h
        local_dir_name = model_name.split('/')[-1]

    local_dir = os.path.join(path, local_dir_name)

    print(f"Downloading model '{model_name}' (revision: {revision})...")
    print(f"Target directory: {local_dir}")

    try:
        # Download the model
        model_path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.safetensors", "*.h5", "*.ot", "*.tflite"]
        )
        print(f"Successfully downloaded model to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def main():
    """Execute the script functionality."""
    args = parse_args()
    download_model(
        model_name=args.model_name,
        path=args.path,
        revision=args.revision,
        local_dir_name=args.local_dir_name
    )


if __name__ == "__main__":
    main()
