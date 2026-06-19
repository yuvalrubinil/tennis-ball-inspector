
import cv2
import numpy as np

class ColorExtractor:

    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        """
        Extracts a normalized 2D Lightness-Blue/Yellow histogram 
        using the CIELAB color space.
        """
        # 1. Move to CIELAB space
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        # 2. Calculate 2D Hist using Channels 0 (L*) and 2 (b*)
        # L* ranges from 0 to 255 in OpenCV; b* also ranges from 0 to 255
        # We use 100 bins for Lightness and 100 bins for the Blue/Yellow axis
        lab_hist = cv2.calcHist([lab_img], [0, 2], mask, [100, 100], [0, 256, 0, 256])
        
        # 3. Normalizing
        cv2.normalize(lab_hist, lab_hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
        
        return lab_hist
    