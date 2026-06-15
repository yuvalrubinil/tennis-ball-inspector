import cv2
import numpy as np
from preprocessor import Preprocessor
from color_extractor import ColorExtractor

def translate_and_crop_char(template_char, char, cropped_ball):
    template_cx, template_cy = template_char['centroid']
    cx, cy = char['centroid']
    d_x = template_cx - cx
    d_y = template_cy - cy

    # translation matrix
    M = np.float32([[1, 0, d_x], [0, 1, d_y]])
    
    h_run, w_run = cropped_ball.shape[:2]
    runtime_mask_uint8 = char['mask'].astype(np.uint8) * 255 
    aligned_mask = cv2.warpAffine(runtime_mask_uint8, M, (w_run, h_run))

    # crop out the exact region of the template's bounding box
    tx, ty, tw, th = template_char['bbox']
    cropped_aligned_char = aligned_mask[ty:ty+th, tx:tx+tw] > 0

    return cropped_aligned_char


class FeatureExtractor:

    def __init__(self, config_path, template_path):
        self.processor = Preprocessor(config_path)
        self.color_extractor = ColorExtractor()

        # getting all the roi
        template_cropped_ball, self.template_r, template_upper_seam_mask, template_lower_seam_mask, template_characters, template_patch = self.processor(template_path)
        
        # extracting all the roi color signatures
        self.template_patch_color_signature = self.color_extractor(template_patch) 
        self.template_upper_seam_color_signature = self.color_extractor(template_cropped_ball, template_upper_seam_mask)
        self.template_lower_seam_color_signature = self.color_extractor(template_cropped_ball, template_lower_seam_mask)

        self.template_characters_color_signatures = []
        self.template_characters = template_characters
        
        # Process template characters dictionary items safely
        for template_char in self.template_characters:
            # color extraction
            char_color_signature = self.color_extractor(template_cropped_ball, template_char['mask'])
            self.template_characters_color_signatures.append(char_color_signature)

            # crop the template mask
            x, y, w, h = template_char['bbox']
            template_char['cropped_mask'] = template_char['mask'][y:y+h, x:x+w].astype(bool)

        
    def __call__(self, image_path):
        # getting all the roi
        cropped_ball, r, upper_seam_mask, lower_seam_mask, characters, patch = self.processor(image_path)
        
        # extracting all the roi color signatures
        patch_color_signature = self.color_extractor(patch)
        upper_seam_color_signature = self.color_extractor(cropped_ball, upper_seam_mask)
        lower_seam_color_signature = self.color_extractor(cropped_ball, lower_seam_mask)
        
        chars_color_signatures = []
        for char in characters:
            char_color_signature = self.color_extractor(cropped_ball, char['mask'])
            chars_color_signatures.append(char_color_signature)

        chars_structural_diff = []
        min_char_count = min(len(self.template_characters), len(characters))
        
        for i in range(min_char_count):
            # calculating char starctual difference
            template_char, char = self.template_characters[i], characters[i]
            char_translated_mask = translate_and_crop_char(template_char, char, cropped_ball)
            template_char_mask = template_char['cropped_mask']

            char_starctual_diff = np.logical_xor(template_char_mask, char_translated_mask)

            # morphological erosion filter
            kernel = np.ones((12, 12), np.uint8)
            char_starctual_diff_filtered = cv2.erode(char_starctual_diff.astype(np.uint8), kernel, iterations=1).astype(bool)
            chars_structural_diff.append(char_starctual_diff_filtered)

        return r, patch_color_signature, upper_seam_color_signature, lower_seam_color_signature, chars_color_signatures, chars_structural_diff

