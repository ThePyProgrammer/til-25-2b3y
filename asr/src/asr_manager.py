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
        self.asr_model.to(torch.float16)

        self.max_batch_size = 16

        # self.asr_model.cfg.decoding.greedy.max_symbols = 5
        # self.asr_model.cfg.decoding.beam.beam_size = 1
        # self.asr_model.cfg.decoding.durations = [0, 1, 2]

    def asr(self, encoded: list[bytes]) -> list[str]:
        """Performs ASR transcription on a batch of audio files.
        Args:
            encoded: A list of audio files in bytes.
        Returns:
            A list of strings containing the transcriptions of each audio file.
        """
        transcriptions = []
        try:
            all_audio_tensors = []
            for audio_bytes in encoded:
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
                audio_tensor = torch.tensor(audio_data, dtype=torch.float16)
                all_audio_tensors.append(audio_tensor)

            # Process in batches of self.max_batch_size
            for i in range(0, len(all_audio_tensors), self.max_batch_size):
                batch_tensors = all_audio_tensors[i:i+self.max_batch_size]
                with torch.no_grad():
                    # Transcribe using the NeMo model in batch
                    batch_transcriptions = self.asr_model.transcribe(batch_tensors)

                # Extract the text from each transcription
                for transcription in batch_transcriptions:
                    if transcription and hasattr(transcription, 'text'):
                        transcriptions.append(transcription.text)
                    else:
                        transcriptions.append("i am steve")
        except Exception as e:
            print(f"Error transcribing audio batch: {e}")
            # In case of error, return empty strings for all inputs
            transcriptions = [""] * len(encoded)
        return transcriptions
