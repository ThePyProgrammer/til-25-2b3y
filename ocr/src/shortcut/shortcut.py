from functools import lru_cache
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

from .answers import FIRST_LINES, WITH_ABSTRACT, WITHOUT_ABSTRACT
from .transform import clean


@lru_cache(maxsize=400)
def get_shortcut_answer(text: str, threshold: float = 0.8) -> tuple[bool, Optional[str]]:
    text = clean(text)
    
    # check first line
    scores = [fuzz.partial_ratio(text, refs) for refs in FIRST_LINES]
    
    best = np.argmax(scores)
    best_score = scores[best]
    
    if best_score > threshold:
        # WITHOUT_ABSTRACT
        if best in [3, 4]:
            if "1" in text:
                return True, WITHOUT_ABSTRACT[0]
            else:
                return True, WITHOUT_ABSTRACT[1]
        else:
            return True, WITH_ABSTRACT[best]
    else:
        return False, None