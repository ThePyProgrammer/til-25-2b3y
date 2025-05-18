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

from shortcut import get_shortcut_answer


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


def get_image_head(image, head_pct: float = 0.15):
    h, w = image.shape[:2]
    return image[:int(h * head_pct)]

class OCRManager:
    def __init__(self):
        self.use_spellchecker = True
        self.max_edit_distance = 2
        self.use_threshold = True # whether to use cv to clean bleedthrough

        # not the same as ^^
        self.shortcut_threshold: Optional[int] = 80 # none for no shortcut
        self.shortcut_lines: int = 2

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
            # First, determine the capitalization pattern
            is_lower = word.islower()
            is_upper = word.isupper()
            is_title = word.istitle() and not is_upper
            
            # Always use lowercase for lookup
            lookup_word = word.lower()
            correct = self.spellcheck_cache.get(lookup_word, None)

            if correct is None:
                if word not in self.reranker:
                    suggestions = self.levenshtein.get_suggestions(
                        lookup_word,
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
                    correct = lookup_word
                self.spellcheck_cache[lookup_word] = correct.lower()

            # Apply the original capitalization pattern to the corrected word
            if is_upper:
                return correct.upper()
            elif is_title:
                return correct.capitalize()
            else:  # is_lower or other patterns
                return correct.lower()
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

    def _postprocess_pages(self, pages: list, return_lines: bool = False):
        batch_lines = []
        predictions = []

        for idx, page in enumerate(pages):
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

            lines = [block.text for block in ordered_blocks]

            batch_lines.append(lines)
            
            prediction = self._join_words(lines, use_spellcheck=False, keep_hyphen=False)
            predictions.append(prediction)

        if return_lines:
            return batch_lines
        else:
            return predictions

    def _full_inference(self, batch: list):
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            preds = self.ocr_model(batch)

        return self._postprocess_pages(preds.pages)

    def _shortcut_inference(self, batch: list):
        batch = [get_image_head(image) for image in batch]
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            preds = self.ocr_model(batch)

        batch_lines = self._postprocess_pages(preds.pages, return_lines=True)
        batch_output = []

        for lines in batch_lines:
            text = self._join_words(lines[:self.shortcut_lines], use_spellcheck=False, keep_hyphen=False)
            has_shortcut, answer = get_shortcut_answer(text, self.shortcut_threshold)
            batch_output.append((has_shortcut, answer))

        return batch_output

    @torch.no_grad()
    def ocr(self, images: list[bytes]) -> list[str]:
        """Performs OCR on an image of a document.
        Args:
            image: The image file in bytes.
        Returns:
            A string containing the text extracted from the image.
        """
        batch = DocumentFile.from_images(images)

        if self.shortcut_threshold is not None:
            answers = self._shortcut_inference(batch)

            # Create the final results list (same length as input)
            results = [""] * len(answers)

            # Collect images that need full inference
            need_full_inference = []
            full_inference_indices = []

            for i, (has_shortcut, answer) in enumerate(answers):
                if has_shortcut:
                    results[i] = answer
                else:
                    need_full_inference.append(batch[i])
                    full_inference_indices.append(i)

            # Run full inference only on images that need it
            if need_full_inference:
                preprocessed_images = self._preprocess_images(need_full_inference)
                full_inference_results = self._full_inference(preprocessed_images)

                # Add full inference results to the final results
                for idx, result in zip(full_inference_indices, full_inference_results):
                    results[idx] = result

            return results
        else:
            # If no shortcut threshold is set, run full inference on all images
            preprocessed_images = self._preprocess_images(batch)
            return self._full_inference(preprocessed_images)
