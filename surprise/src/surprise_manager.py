import io
from PIL import Image, ImageFilter
import numpy as np
from numpy import dot
import cv2
from numpy.linalg import norm

from skimage.metrics import structural_similarity as ssim

def edge_ssim(a, b):
    return ssim(a[:, np.newaxis], b[:, np.newaxis], data_range=b.max() - b.min())

class SurpriseManager:
    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        pass

    def surprise(self, slices: list[bytes]) -> list[int]:
        """
        Reconstructs shredded document from vertical slices.

        Args:
            slices: list of byte arrays, each representing a JPEG-encoded vertical slice of the input document

        Returns:
            Predicted permutation of input slices to correctly reassemble the document.
        """

        # Your inference code goes here
        i = [Image.open(io.BytesIO(image_bytes)).convert("L") for image_bytes in slices]
        i_np = [np.array(im) for im in i]
        
        ords = []
        existing = list(range(len(slices)))
        
        first_img = -1
        last_img = -1
        
        # EDGE_WIDTH = 1
        
        left_edges = np.array([np.array(img)[:, 0] for img in i_np], dtype=np.float64)
        right_edges = np.array([np.array(img)[:, -1] for img in i_np], dtype=np.float64)

        for k in range(len(slices)):
            if first_img == -1 and np.all(left_edges[k] == 255):
                first_img = k
                existing.remove(k)
            elif last_img == -1 and np.all(right_edges[k] == 255):
                last_img = k
                existing.remove(k)
        
        
        if first_img == -1 and last_img == -1:
            # edge case where it didn't find a suitable sample for first or last
            existing.remove(0)
            ords.append(0)
            # first_img = 0
        
        if first_img != -1:
            ords.append(first_img)
        elif last_img != -1:
            # edge case where it didn't find a suitable first sample
            ords.append(last_img)
        
        while len(existing) != 0:
            if first_img != -1:
                # normal case, first_img is defined, last_img may or may not be
                if len(existing) == 1:
                    min_idx = 0
                else:
                    curr_img = cv2.GaussianBlur(right_edges[[ords[-1]]], (1, 3), sigmaX=0)
                    images = left_edges[existing]
                    
                    diffs = np.square(images-curr_img).sum(1)
                    
#                     diffs = []
#                     for idx in existing:
#                         new_img = left_edges[idx]
#                         diff = np.square(new_img - curr_img).sum()
#                         diffs.append(diff)
                    
#                     diffs = np.array(diffs)

                    # find the index stored in existing that minimized the value
                    min_idx = diffs.argmin()
                popped_el = existing.pop(min_idx)
                ords.append(popped_el)
            elif last_img != -1:
                # edge case where it didn't find a suitable first sample
                if len(existing) == 1:
                    min_idx = 0
                else:
                    curr_img = cv2.GaussianBlur(left_edges[[ords[0]]], (1, 3), sigmaX=0)
                    images = right_edges[existing]
                    
                    diffs = np.square(images-curr_img).sum(1)
                    
                    # diffs = ssim(images, curr_img)
                    
#                     diffs = []
#                     for idx in existing:
#                         new_img = right_edges[idx]
#                         diff = np.square(new_img - curr_img).sum()
#                         diffs.append(diff)
                    
#                     diffs = np.array(diffs)
                    
                    # find the index stored in existing that minimized the value
                    min_idx = diffs.argmin()
                popped_el = existing.pop(min_idx)
                ords.insert(0, popped_el)
            else:
                # last case, it couldn't find a first or last edge so it's just gonna piece itself together.
                curr_img_l = cv2.GaussianBlur(left_edges[[ords[0]]], (1, 3), sigmaX=0)
                images_l = right_edges[existing]
                curr_img_r = cv2.GaussianBlur(right_edges[[ords[-1]]], (1, 3), sigmaX=0)
                images_r = left_edges[existing]
                
                diffs_l = np.square(images_l-curr_img_l).sum(1)
                diffs_r = np.square(images_r-curr_img_r).sum(1)
                min_l,lx = diffs_l.min(), diffs_l.argmin()
                min_r, rx = diffs_r.min(), diffs_r.argmin()
                
                if min_l > min_r:
                    popped_el = existing.pop(rx)
                    ords.append(popped_el)
                else:
                    popped_el = existing.pop(lx)
                    ords.insert(0, popped_el)
        
        if first_img != -1 and last_img != -1:
            # edge case where they found both
            ords.append(last_img)
        
        return ords