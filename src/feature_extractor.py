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


def hist_diff(h1, h2, bins=100):
    """True 2D Earth Mover's Distance between two (bins,bins) L1-normalized histograms."""
    # Build (value, x, y) signature arrays required by cv2.EMD
    def to_signature(h):
        idx = np.argwhere(h > 0)
        weights = h[idx[:, 0], idx[:, 1]].astype(np.float32)
        sig = np.column_stack([weights, idx[:, 1].astype(np.float32), idx[:, 0].astype(np.float32)])
        return sig.astype(np.float32)

    sig1 = to_signature(h1)
    sig2 = to_signature(h2)
    dist, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    max_dist = np.sqrt(2) * (bins - 1)  # = true worst-case EMD when total mass = 1.0
    return float(np.clip(dist / max_dist, 0.0, 1.0))

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

        r_ratio = r / self.template_r
        
        # extracting all the roi color signatures
        patch_color_signature = self.color_extractor(patch)
        upper_seam_color_signature = self.color_extractor(cropped_ball, upper_seam_mask)
        lower_seam_color_signature = self.color_extractor(cropped_ball, lower_seam_mask)

        # calculating color signature differences
        patch_color_diff = hist_diff(self.template_patch_color_signature, patch_color_signature)
        upper_seam_color_diff = hist_diff(self.template_upper_seam_color_signature, upper_seam_color_signature)
        lower_seam_color_diff = hist_diff(self.template_lower_seam_color_signature, lower_seam_color_signature)
        
        chars_structural_diff = []
        chars_color_diff = []
        min_char_count = min(len(self.template_characters), len(characters))
        valid_char_count = len(self.template_characters) == len(characters)
        for i in range(min_char_count):
            template_char, char = self.template_characters[i], characters[i]

            # calculating color signature difference
            char_color_signature = self.color_extractor(cropped_ball, char['mask'])
            color_diff = hist_diff(self.template_characters_color_signatures[i], char_color_signature)
            chars_color_diff.append(color_diff)

            # calculating char starctual difference
            char_translated_mask = translate_and_crop_char(template_char, char, cropped_ball)
            template_char_mask = template_char['cropped_mask']
            char_starctual_diff_map = np.logical_xor(template_char_mask, char_translated_mask)

            # morphological erosion filter
            kernel = np.ones((12, 12), np.uint8)
            char_starctual_diff_filtered = cv2.erode(char_starctual_diff_map.astype(np.uint8), kernel, iterations=1).astype(bool)
            
            # normalizing
            char_structural_diff = np.sum(char_starctual_diff_filtered) / char_starctual_diff_filtered.size
            chars_structural_diff.append(char_structural_diff)

        return r_ratio, patch_color_diff, upper_seam_color_diff, lower_seam_color_diff, valid_char_count, chars_color_diff, chars_structural_diff
    

    def visualized_features(self, image_path):
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

        chars_cropped = []
        min_char_count = min(len(self.template_characters), len(characters))
        valid_char_count = len(self.template_characters) == len(characters)
        for i in range(min_char_count):
            # calculating char starctual difference
            template_char, char = self.template_characters[i], characters[i]
            char_translated_mask = translate_and_crop_char(template_char, char, cropped_ball)
            chars_cropped.append(char_translated_mask)

        return r, patch_color_signature, upper_seam_color_signature, lower_seam_color_signature, valid_char_count, chars_color_signatures, chars_cropped
    

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
