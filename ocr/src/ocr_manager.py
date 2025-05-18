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

class LPWrapper:
    def __init__(self):
        self.model = lp.models.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
            label_map   = {0: "Text", 1: "Title"}, # In model`label_map`
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
            device="cuda"
        )
    
    def __call__(self, image):
        return self.model.detect(image)
        
class OCRManager:
    def __init__(self):
        self.use_spellchecker = True
        self.max_edit_distance = 2
        self.use_layout_parser = True # 0.6 -> 0.8 when True
        self.use_dilation = False # cleans bleedthrough
        
        det_model = linknet_resnet18(pretrained=False, pretrained_backbone=False)
        reco_model = crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False)
        
        det_params = torch.load("models/linknet_resnet18_ft.pt", map_location="cpu")
        det_model.load_state_dict(det_params)
        
        reco_params = torch.load("models/crnn_mobilenet_v3_small_ft.pt", map_location="cpu")
        reco_model.load_state_dict(reco_params)
        
        self.ocr_model = ocr_predictor(det_model, reco_model, pretrained=False).eval().cuda().half()
        
        if self.use_layout_parser:
            self.lp_model = LPWrapper()
        
        # self.ocr_model.to(device='cuda', dtype=torch.float16)
        
        self.reranker: SpellChecker = SpellChecker()
        self.levenshtein: Levenshtein = Levenshtein()
        self.reranker.word_frequency.load_json("word_frequency.json")
        self.levenshtein.add_from_path("words.txt")

        self.spellcheck_cache: dict[str, str] = {}
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def _dilate(self, batch):
        if self.use_dilation:
            output = []
            for image in batch:
                dilated_image = cv2.dilate(image, self.kernel, iterations=1)
                output.append(dilated_image)
            return output
        else:
            return batch
    
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
    
    @torch.no_grad()
    def ocr(self, images: list[bytes]) -> list[str]:
        """Performs OCR on an image of a document.
        Args:
            image: The image file in bytes.
        Returns:
            A string containing the text extracted from the image.
        """
        batch = DocumentFile.from_images(images)
        
        batch = self._dilate(batch)
        
        if self.use_layout_parser:
            batch_text_blocks = self._process_layout_batch(batch)
        
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            # Run OCR model on the batch
            ocr_results = self.ocr_model(batch)
            
        batch_ordered_lines = []

        for idx, page in enumerate(ocr_results.pages):
            if self.use_layout_parser:
                text_blocks = batch_text_blocks[idx]
                ordered_lines = self._order_lines_using_layout(page, text_blocks)
            else:
                ordered_lines = self._order_lines_using_columns(page)

            batch_ordered_lines.append(ordered_lines)

        output = self._process_text_from_lines(batch_ordered_lines)
        return output

    def _process_layout_batch(self, batch):
        """Process a batch of images to extract layout information."""
        batch_text_blocks = []
        for image in batch:
            ordered_blocks = self._extract_text_blocks_from_image(image)
            batch_text_blocks.append(ordered_blocks)
        return batch_text_blocks

    def _extract_text_blocks_from_image(self, image):
        """Extract and order text blocks from an image."""
        # Get layout detection using layoutparser
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.float16):
            layout = self.lp_model(image)
        text_blocks = lp.Layout([b for b in layout if b.type=='Text' or b.type=="Title"])

        # Deduplicate overlapping text blocks
        text_blocks = self._deduplicate_text_blocks(text_blocks)

        h, w = image.shape[:2]

        # Separate left and right blocks
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

        # Get right blocks and sort them top-to-bottom
        right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
        right_blocks.sort(key=lambda b: b.coordinates[1], inplace=True)

        # Combine left and right blocks with ID assignments
        ordered_blocks = lp.Layout([b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)])
        return ordered_blocks
    
    def _deduplicate_text_blocks(self, text_blocks):
        """
        Deduplicate text blocks by removing larger blocks that significantly overlap with smaller ones.

        Args:
            text_blocks: The layout parser Layout object containing text blocks

        Returns:
            A new Layout object with duplicates removed
        """
        if len(text_blocks) <= 1:
            return text_blocks

        # Calculate area for each block and store in a dictionary
        block_areas = {}
        for i, block in enumerate(text_blocks):
            x1, y1, x2, y2 = block.coordinates
            block_areas[i] = (x2-x1)*(y2-y1)

        # Sort blocks by area (smallest to largest)
        sorted_indices = sorted(range(len(text_blocks)), key=lambda i: block_areas[i])

        # Keep track of blocks to keep
        blocks_to_keep = []

        # Compare each block with other blocks
        for idx in sorted_indices:
            small_block = text_blocks[idx]
            should_keep = True
            x1_s, y1_s, x2_s, y2_s = small_block.coordinates
            small_block_area = block_areas[idx]

            # Check this block against all larger blocks we've decided to keep
            for kept_block in blocks_to_keep:
                x1_k, y1_k, x2_k, y2_k = kept_block.coordinates

                # Calculate intersection area
                x_overlap = max(0, min(x2_s, x2_k) - max(x1_s, x1_k))
                y_overlap = max(0, min(y2_s, y2_k) - max(y1_s, y1_k))
                overlap_area = x_overlap * y_overlap

                # Calculate percentage of small block covered by kept block
                overlap_percentage = overlap_area / small_block_area if small_block_area > 0 else 0

                # If significant overlap (>80%) with an already kept block, skip this one
                if overlap_percentage > 0.9:
                    should_keep = False
                    break

            if should_keep:
                blocks_to_keep.append(small_block)

        # Create a new Layout with the blocks to keep
        return lp.Layout(blocks_to_keep)

    def _order_lines_using_layout(self, page, text_blocks):
        """Order lines using layout block information."""
        h, w = page.dimensions
        ordered_lines = []

        # Extract all lines from the page
        all_lines = self._extract_all_lines(page)

        # For each layout block, find the lines that belong to it
        for layout_block in text_blocks:
            block_lines = []
            # Get block coordinates
            block_x1, block_y1, block_x2, block_y2 = layout_block.coordinates
            block_x1 /= w
            block_y1 /= h
            block_x2 /= w
            block_y2 /= h

            # Find all lines with center point within this block
            for line in all_lines:
                # Calculate center point of the line
                # line.geometry is a tuple of tuples ((x1,y1), (x2,y2))
                points = line.geometry
                center_x = (points[0][0] + points[1][0]) / 2
                center_y = (points[0][1] + points[1][1]) / 2

                # Check if center point is within the current block
                if (center_x >= block_x1 and center_x <= block_x2 and
                    center_y >= block_y1 and center_y <= block_y2):
                    block_lines.append(line)

            # Sort lines within block by height (y-coordinate)
            block_lines.sort(key=lambda line: (line.geometry[0][1] + line.geometry[1][1]) / 2)
            ordered_lines.extend(block_lines)

        return ordered_lines

    def _order_lines_using_columns(self, page):
        """Order lines by separating them into left and right columns."""
        h, w = page.dimensions

        # Extract all lines from the page
        all_lines = self._extract_all_lines(page)

        # Separate left and right lines
        left_lines = []
        right_lines = []

        for line in all_lines:
            points = line.geometry
            center_x = (points[0][0] + points[1][0]) / 2

            # Check if center point is in left interval
            if center_x <= w/2*1.05:
                left_lines.append(line)
            else:
                right_lines.append(line)

        # Sort each column by vertical position
        left_lines.sort(key=lambda line: (line.geometry[0][1] + line.geometry[1][1]) / 2)
        right_lines.sort(key=lambda line: (line.geometry[0][1] + line.geometry[1][1]) / 2)

        # Combine left and right lines
        ordered_lines = left_lines + right_lines
        return ordered_lines

    def _extract_all_lines(self, page):
        """Extract all lines from a page."""
        all_lines = []
        for block in page.blocks:
            all_lines.extend(block.lines)
        return all_lines

    def _process_text_from_lines(self, batch_ordered_lines):
        """Process text from ordered lines."""
        output = []
        for lines in batch_ordered_lines:
            joined = ""
            for line in lines:
                for word in line.words:
                    word = word.value
                    if word.endswith('-') and len(word) > 1:
                        new = word[:-1]
                        new = self._spellcheck(new)
                    else:
                        word = self._spellcheck(word)
                        new = f"{word} "
                    joined += new
            output.append(joined)
        return output