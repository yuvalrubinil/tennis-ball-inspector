import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from src.feature_extractor import FeatureExtractor

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def color_line(text, is_valid):
    return f"{GREEN}{text}{RESET}" if is_valid else f"{RED}{text}{RESET}"

class BallInspector:

    def __init__(self, config_path, template_path):
        self.feature_extractor = FeatureExtractor(config_path, template_path)
        self.template_maps = [
            None,
            self.feature_extractor.template_patch_color_signature,
            self.feature_extractor.template_upper_seam_color_signature,
            self.feature_extractor.template_lower_seam_color_signature,
            None,
            self.feature_extractor.template_characters_color_signatures,
            None
            ]
        with open(config_path, 'r') as config:
            config = json.load(config)["thresholds"]
            self.radius_tolarence = config["radius_tolarence"]
            self.patch_color_thresh = config["patch_color"]
            self.seam_lines_color_thresh = config["seam_lines_color"]
            self.char_color_thresh = config["char_color"]
            self.char_structur_thresh = config["char_structur"]


    def __call__(self, image_path, visualize_maps=False):

        feature_vector, visualization = self.feature_extractor(image_path, visualize=visualize_maps)
        r_ratio, patch_color_diff, upper_seam_color_diff, lower_seam_color_diff, valid_char_count, chars_color_diff, chars_structural_diff = feature_vector

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
        
        if visualize_maps:
            self.visualize_maps(visualization)


    
    def visualize_maps(self,runtime_outputs):
        """Visualizes all color signatures and structural maps"""
        (_, t_patch, t_upper, t_lower, _, t_chars, _) = self.template_maps
        (_, r_patch, r_upper, r_lower, _, r_chars, r_struct,) = runtime_outputs

        num_chars = min(len(t_chars), len(r_chars))
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

        gs_global = gs[0].subgridspec(1, 3, wspace=0.3)
        global_items = [
            (t_patch, r_patch, "Patch"),
            (t_upper, r_upper, "Upper Seam"),
            (t_lower, r_lower, "Lower Seam"),
        ]

        for idx, (t_sig, r_sig, name) in enumerate(global_items):
            gs_pair = gs_global[idx].subgridspec(1, 2, wspace=0.05)
            
            ax_t = fig.add_subplot(gs_pair[0])
            ax_t.imshow(t_sig, cmap="jet", aspect="auto")
            ax_t.set_title(f"Template {name}")
            ax_t.axis("off")

            ax_r = fig.add_subplot(gs_pair[1])
            ax_r.imshow(r_sig, cmap="jet", aspect="auto")
            ax_r.set_title(f"Runtime {name}")
            ax_r.axis("off")

        if num_chars > 0:
            gs_chars = gs[1].subgridspec(1, num_chars, wspace=0.3)
            for i in range(num_chars):
                gs_pair = gs_chars[i].subgridspec(1, 2, wspace=0.05)

                ax_t = fig.add_subplot(gs_pair[0])
                ax_t.imshow(t_chars[i], cmap="jet", aspect="auto")
                ax_t.set_title(f"T Char {i + 1}")
                ax_t.axis("off")

                ax_r = fig.add_subplot(gs_pair[1])
                ax_r.imshow(r_chars[i], cmap="jet", aspect="auto")
                ax_r.set_title(f"R Char {i + 1}")
                ax_r.axis("off")

        if len(r_struct) > 0:
            gs_struct = gs[2].subgridspec(1, len(r_struct), wspace=0.3)
            for i, diff_map in enumerate(r_struct):
                ax = fig.add_subplot(gs_struct[i])
                ax.imshow(diff_map.astype(np.uint8), cmap="hot", aspect="equal")
                diff_pixels = np.count_nonzero(diff_map)
                ax.set_title(f"Char {i + 1}\nPixels={diff_pixels}")
                ax.axis("off")
        else:
            ax = fig.add_subplot(gs[2])
            ax.text(0.5, 0.5, "No character structural maps available", ha="center", va="center", fontsize=14)
            ax.axis("off")

        plt.suptitle("Tennis Ball Inspection Dashboard", fontsize=16, fontweight="bold")
        plt.show()

        
        
