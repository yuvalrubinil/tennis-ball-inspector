import cv2
import json
from feature_extractor import FeatureExtractor

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def color_line(text, is_valid):
    return f"{GREEN}{text}{RESET}" if is_valid else f"{RED}{text}{RESET}"

class VerdictHead:

    def __init__(self, config_path, template_path):
        self.feature_extractor = FeatureExtractor(config_path, template_path)
        with open(config_path, 'r') as config:
            config = json.load(config)["thresholds"]
            self.radius_tolarence = config["radius_tolarence"]
            self.patch_color_thresh = config["patch_color"]
            self.seam_lines_color_thresh = config["seam_lines_color"]
            self.char_color_thresh = config["char_color"]
            self.char_structur_thresh = config["char_structur"]


    def __call__(self, image_path):
        r_ratio, patch_color_diff, upper_seam_color_diff, lower_seam_color_diff, valid_char_count, chars_color_diff, chars_structural_diff = self.feature_extractor(image_path)

        r_min, r_max = self.radius_tolarence
        print(color_line(f"Radius Ratio          : {r_ratio:.4f}", r_min <= r_ratio and r_ratio <= r_max))

        print(color_line(f"Patch Color Diff      : {patch_color_diff:.6f}", patch_color_diff <= self.patch_color_thresh))

        avg_seam_color_diff = 0.5 * (upper_seam_color_diff + lower_seam_color_diff)
        print(color_line(f"Seam Color Diff       : {avg_seam_color_diff:.6f}", avg_seam_color_diff <= self.seam_lines_color_thresh))

        print(color_line(f"Valid Char Count      : {valid_char_count}", valid_char_count))

        print("\nCharacters:")
        for i in range(len(chars_color_diff)):
            color_diff = chars_color_diff[i]
            structural_diff = chars_structural_diff[i]

            is_char_color_valid = color_diff <= self.char_color_thresh
            is_char_struct_valid = structural_diff <= self.char_structur_thresh

            color_str = f"Color={color_diff:.6f}"
            color_part = f"{GREEN}{color_str}{RESET}" if is_char_color_valid else f"{RED}{color_str}{RESET}"

            struct_str = f"Structure={structural_diff:.6f}"
            struct_part = f"{GREEN}{struct_str}{RESET}" if is_char_struct_valid else f"{RED}{struct_str}{RESET}"

            print(f"  Char {i + 1}: {color_part}, {struct_part}")

        print("=" * 60)
        
        
