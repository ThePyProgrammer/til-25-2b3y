"""Manages the ASR model."""

import io
import os
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from scipy.io import wavfile
from noisereduce.torchgate import TorchGate as TG
from spellchecker import SpellChecker
from spellwise import Levenshtein
import string

class ASRManager:

    def __init__(self):
        self.use_spellchecker = False
        self.use_noise_reduction = False
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        model_name = "parakeet-tdt-0.6b-v2"
        print(f"Loading ASR model '{model_name}'")
        # self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.asr_model = nemo_asr.models.ASRModel.restore_from(f"./models/{model_name}/{model_name}.nemo")
        self.asr_model.half().cuda()
        
        self.gate = TG(sr=16000, nonstationary=True).half().cuda()

        self.max_batch_size = 16
        
        self.reranker: SpellChecker = SpellChecker()
        self.levenshtein: Levenshtein = Levenshtein()
        self.reranker.word_frequency.load_json("word_frequency.json")
        self.levenshtein.add_from_path("words.txt")
        self.spellcheck_cache: dict[str, str] = {}

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
                audio_tensor = torch.tensor(audio_data).half().cuda()
                if self.use_noise_reduction:
                    audio_tensor = self.gate(audio_tensor.unsqueeze(0)).squeeze(0)
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
                        text = " ".join([self._spellcheck(word) for word in transcription.text.split(" ")])
                        transcriptions.append(text)
                    else:
                        transcriptions.append("i am steve")
        except Exception as e:
            print(f"Error transcribing audio batch: {e}")
            # In case of error, return empty strings for all inputs
            transcriptions = [""] * len(encoded)
        return transcriptions

    def _spellcheck(self, word: str) -> str:
        if self.use_spellchecker:
            # First, determine the capitalization pattern
            is_lower = word.islower()
            is_upper = word.isupper()
            is_title = word.istitle() and not is_upper
            punctuation = word[-1] if word[-1] in string.punctuation else ""
            
            # Always use lowercase for lookup
            lookup_word = word.lower()
            correct = self.spellcheck_cache.get(lookup_word, None)

            if correct is None:
                if word not in self.reranker:
                    suggestions = self.levenshtein.get_suggestions(
                        lookup_word,
                        max_distance=2
                    )
                    if len(suggestions) > 0:
                        # Sort suggestions by distance
                        suggestions.sort(key=lambda x: x['distance'])
                        # Find all suggestions with the minimum distance
                        min_distance = suggestions[0]['distance']
                        best_suggestions = [s for s in suggestions if s['distance'] == min_distance]
                        # Rerank by word frequency
                        best_word = max(
                            best_suggestions,
                            key=lambda s: self.reranker.word_frequency.dictionary.get(s['word'], 0)
                        )['word']
                        correct = best_word
                if correct is None:
                    correct = lookup_word
                self.spellcheck_cache[lookup_word] = correct.lower()

            # Apply the original capitalization pattern to the corrected word
            if punctuation and correct[-1] not in string.punctuation:
                correct += punctuation
            if is_upper:
                return correct.upper()
            elif is_title:
                return correct.capitalize()
            else:  # is_lower or other patterns
                return correct.lower()
            
        else:
            return word