"""Manages the OCR model."""

import torch
import cv2

from doctr.models import ocr_predictor
from doctr.models.detection import linknet_resnet18
from doctr.models.recognition import crnn_mobilenet_v3_small
from doctr.io import DocumentFile

import layoutparser as lp

from spellchecker import SpellChecker
from spellwise import Levenshtein


detection_arch = [
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
    "fast_tiny",
    "fast_small",
    "fast_base",
]

recognition_arch = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]
        
class OCRManager:
    def __init__(self):
        self.use_spellchecker = True
        self.max_edit_distance = 2
        self.use_threshold = True # cleans bleedthrough
        
        det_model = linknet_resnet18(pretrained=False, pretrained_backbone=False)
        reco_model = crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False)
        
        det_params = torch.load("models/linknet_resnet18_ft.pt", map_location="cpu")
        det_model.load_state_dict(det_params)
        
        reco_params = torch.load("models/crnn_mobilenet_v3_small_ft.pt", map_location="cpu")
        reco_model.load_state_dict(reco_params)
        
        self.ocr_model = ocr_predictor(det_model, reco_model, pretrained=False).eval().cuda().half()
        
        self.reranker: SpellChecker = SpellChecker()
        self.levenshtein: Levenshtein = Levenshtein()
        self.reranker.word_frequency.load_json("word_frequency.json")
        self.levenshtein.add_from_path("words.txt")

        self.spellcheck_cache: dict[str, str] = {}
        
    def _preprocess_images(self, batch):
        """Apply dilation to enhance image if enabled"""
        output = []

        for image in batch:
            # Ensure image is grayscale (8-bit, single channel)
            if len(image.shape) > 2:  # If image has more than 2 dimensions, it's not grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            if self.use_threshold:
                mask = cv2.threshold(
                    gray_image, 190, 255,
                    cv2.THRESH_BINARY,
                )[-1] == 1

                gray_image = gray_image | mask 

            normal_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            output.append(normal_image)

        return output
    
    def _spellcheck(self, word: str) -> str:
        if self.use_spellchecker:
            correct = self.spellcheck_cache.get(word.lower(), None)

            if correct is None:
                if word not in self.reranker:
                    suggestions = self.levenshtein.get_suggestions(
                        word,
                        max_distance=self.max_edit_distance
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
                    correct = word
                self.spellcheck_cache[word.lower()] = correct

            return correct
        else:
            return word
    
    def _join_words(self, words: list[str], keep_hyphen=True, use_spellcheck=True):
        joined = ""
        for word in words:
            if word.endswith('-') and len(word) > 1:
                new = word[:-1]
                if use_spellcheck:
                    new = self._spellcheck(new)
                if keep_hyphen:
                    new += "-"
            else:
                if use_spellcheck:
                    word = self._spellcheck(word)
                new = f"{word} "
            joined += new
        return joined.strip()
    
    @torch.no_grad()
    def ocr(self, images: list[bytes]) -> list[str]:
        """Performs OCR on an image of a document.
        Args:
            image: The image file in bytes.
        Returns:
            A string containing the text extracted from the image.
        """
        batch = DocumentFile.from_images(images)
        
        # batch = self._preprocess_images(batch)
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            ocr_results = self.ocr_model(batch)

        batch_text_blocks = []
        predictions = []

        for idx, page in enumerate(ocr_results.pages):
            text_blocks = []

            h, w = page.dimensions

            for line in page.blocks[0].lines:

                rect = lp.Rectangle(
                    line.geometry[0][0] * w,
                    line.geometry[0][1] * h,
                    line.geometry[1][0] * w,
                    line.geometry[1][1] * h,
                )
                words = [word.value for word in line.words]
                text = self._join_words(words, use_spellcheck=True, keep_hyphen=True)
                text_block = lp.TextBlock(rect, text=text)

                text_blocks.append(text_block)

            text_blocks = lp.Layout(text_blocks)

            # Separate left and right blocks
            left_interval = lp.Interval(0, w/2*1.05, axis='x')
            left_blocks = text_blocks.filter_by(left_interval, center=True)
            left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

            # Get right blocks and sort them top-to-bottom
            right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
            right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

            # Combine left and right blocks with ID assignments
            ordered_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])

            batch_text_blocks.append(ordered_blocks)

            lines = [block.text for block in ordered_blocks]
            prediction = self._join_words(lines, use_spellcheck=False, keep_hyphen=False)

            predictions.append(prediction)
        
        return predictions