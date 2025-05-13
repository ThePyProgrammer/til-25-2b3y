#!/usr/bin/env python3
"""
Setup script to preload NeMo ASR model at build time.
This ensures the model is available during inference even without internet access.
"""

import nemo.collections.asr as nemo_asr

def main():
    """
    Downloads and caches the NeMo ASR model.
    """
    print("Starting NeMo ASR model preloading...")

    # Preload and cache the model
    try:
        print("Downloading and caching NeMo ASR model...")
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        _ = nemo_asr.models.ASRModel.from_pretrained(model_name)

        print(f"Successfully preloaded model: {model_name}")
    except Exception as e:
        print(f"Error preloading model: {e}")
        raise

    print("ASR model setup completed successfully!")

if __name__ == "__main__":
    main()
