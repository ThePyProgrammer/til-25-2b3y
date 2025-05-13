"""Manages the ASR model."""

import io
import os
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from scipy.io import wavfile


class ASRManager:

    def __init__(self):
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
        print(f"Loading ASR model '{model_name}'")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file.

        Args:
            audio_bytes: The audio file in bytes.

        Returns:
            A string containing the transcription of the audio.
        """
        try:
            # Convert bytes to in-memory file object
            audio_file = io.BytesIO(audio_bytes)

            # Read the WAV file
            sample_rate, audio_data = wavfile.read(audio_file)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Ensure the audio data is in the right format (float32)
            if audio_data.dtype != np.float32:
                # Normalize the int audio data to float
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)

            # Convert to tensor
            audio_tensor = torch.tensor(audio_data).unsqueeze(0)

            # Transcribe using the NeMo model
            transcriptions = self.asr_model.transcribe([audio_tensor])

            # Return the transcription
            if transcriptions and len(transcriptions) > 0:
                return transcriptions[0]
            else:
                return ""

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""
