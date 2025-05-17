"""Manages the OCR model."""

from doctr.models import ocr_predictor
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
        self.ocr_model = ocr_predictor(detection_arch[-3], recognition_arch[1], pretrained=True)

        self.lp_model = lp.models.Detectron2LayoutModel(
                    config_path ='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', # In model catalog
                    label_map   = {0: "Text", 1: "Title"}, # In model`label_map`
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7] # Optional
                )

        self.use_spellchecker = True

        self.reranker: SpellChecker = SpellChecker()
        self.levenshtein: Levenshtein = Levenshtein()
        self.reranker.word_frequency.load_json("word_frequency.json")
        self.levenshtein.add_from_path("words.txt")

        self.spellcheck_cache: dict[str, str] = {}

    def _spellcheck(self, word: str) -> str:
        if self.use_spellchecker:
            correct = self.spellcheck_cache.get(word, None)
            
            if correct is None:
                if word not in self.reranker:
                    suggestions = self.levenshtein.get_suggestions(word)
                    if len(suggestions) > 0:
                        correct = suggestions[0]['word']
                if correct is None:
                    correct = word
                self.spellcheck_cache[word] = correct
                
            return correct
        else:
            return word

    def ocr(self, images: list[bytes]) -> list[str]:
        """Performs OCR on an image of a document.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the text extracted from the image.
        """

        batch = DocumentFile.from_images(images)

        batch_text_blocks = []

        for image in batch:
            # Get layout detection using layoutparser
            layout = self.lp_model.detect(image)

            text_blocks = lp.Layout([b for b in layout if b.type=='Text' or b.type=="Title"])
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
            batch_text_blocks.append(ordered_blocks)

        # Run OCR model on the batch
        ocr_results = self.ocr_model(batch)

        batch_ordered_lines = []

        for idx, page in enumerate(ocr_results.pages):
            h, w = page.dimensions

            text_blocks = batch_text_blocks[idx]
            ordered_lines = []

            # Extract all lines from the page
            all_lines = []
            for block in page.blocks:
                all_lines.extend(block.lines)

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

            batch_ordered_lines.append(ordered_lines)

        output: list[str] = []

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
