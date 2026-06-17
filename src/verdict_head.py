"""
def __call__(self, image_path):
        # getting all the roi
        cropped_ball, r, upper_seam_mask, lower_seam_mask, characters, patch = self.processor(image_path)
        
        # extracting all the roi color signatures
        patch_color_signature = self.color_extractor(patch)
        upper_seam_color_signature = self.color_extractor(cropped_ball, upper_seam_mask)
        lower_seam_color_signature = self.color_extractor(cropped_ball, lower_seam_mask)
        
        characters_color_signatures = []
        for char in characters:
            char_color_signature = self.color_extractor(cropped_ball, char['mask'])
            characters_color_signatures.append(char_color_signature)

        # calculating color diffs
        patch_color_diff = cv2.compareHist(self.template_patch_color_signature, patch_color_signature, cv2.HISTCMP_BHATTACHARYYA)
        upper_seam_color_diff = cv2.compareHist(self.template_upper_seam_color_signature, upper_seam_color_signature, cv2.HISTCMP_BHATTACHARYYA)
        lower_seam_color_diff = cv2.compareHist(self.template_lower_seam_color_signature, lower_seam_color_signature, cv2.HISTCMP_BHATTACHARYYA)

        chars_color_diff = []
        chars_structural_diff = []
        min_char_count = min(len(self.template_characters), len(characters))
        
        for i in range(min_char_count):
            # calculating char color difference
            color_diff = cv2.compareHist(self.template_characters_color_signatures[i], characters_color_signatures[i], cv2.HISTCMP_BHATTACHARYYA)
            chars_color_diff.append(color_diff)

            # calculating char starctual difference
            template_char, char = self.template_characters[i], characters[i]
            char_translated_mask = translate_and_crop_char(template_char, char, cropped_ball)
            template_char_mask = template_char['cropped_mask']

            char_starctual_diff = np.logical_xor(template_char_mask, char_translated_mask)

            # morphological erosion filter
            kernel = np.ones((12, 12), np.uint8)
            char_starctual_diff_filtered = cv2.erode(char_starctual_diff.astype(np.uint8), kernel, iterations=1).astype(bool)
            chars_structural_diff.append(char_starctual_diff_filtered)

        return patch_color_diff, upper_seam_color_diff, lower_seam_color_diff, chars_color_diff, chars_structural_diff

"""
import cv2
import json
from feature_extractor import FeatureExtractor




class VerdictHead:

    def __init__(self, ball_config_path, template_path, verdcit_config_path):
        self.feature_extractor = FeatureExtractor(ball_config_path, template_path)
        with open(verdcit_config_path, 'r') as config:
            self.config = json.load(config)

    def __call__(self, image_path):
        r_ratio, patch_color_diff, upper_seam_color_diff, lower_seam_color_diff, valid_char_count, chars_color_diff, chars_structural_diff = self.feature_extractor(image_path)
        
        
        