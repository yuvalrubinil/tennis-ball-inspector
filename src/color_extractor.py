
import cv2
import numpy as np

class ColorExtractor:

    def __init__(self):
        pass

    def __call__(self, image, mask=None):
        """
        Extracts a normalized 2D lightness-blue/yellow histogram 
        using the CIELAB color space.
        """
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        # using lightness and blue/yellow axis due to the tennis ball colors
        # using 100 bins for lightness and 100 bins for the blue/yellow axis
        lab_hist = cv2.calcHist([lab_img], [0, 2], mask, [100, 100], [0, 256, 0, 256])
        
        # normalizing
        cv2.normalize(lab_hist, lab_hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
        
        return lab_hist
    