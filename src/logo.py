import cv2
import numpy as np
import matplotlib.pyplot as plt
from processor import Processor

def compute_masked_hsv_histogram_distance(patch_a, mask_a, patch_b, mask_b):
    """
    Computes the Bhattacharyya distance between the 2D H-S histograms 
    of two patches, evaluated ONLY where their respective binary masks are white (255).
    """
    hsv_a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2HSV)
    
    # Calculate 2D Histograms utilizing the specific character binary masks
    hist_a = cv2.calcHist([hsv_a], [0, 1], mask_a, [128, 128], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], mask_b, [128, 128], [0, 180, 0, 256])
    
    # Store log distribution for better heatmap contrast during plotting
    hist_a_plot = np.log1p(hist_a)
    hist_b_plot = np.log1p(hist_b)
    
    # Normalize for Bhattacharyya distance matching
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    
    return distance, hist_a_plot, hist_b_plot


def evaluate_and_plot_character_comparison(patch_a, mask_a, patch_b, mask_b, label_a="Img A", label_b="Img B", title_tag="Character"):
    """
    Isolates white mask pixels of the characters on the underlying BGR images, 
    calculates color distance metrics, and plots the side-by-side dashboard.
    """
    distance, hist_a_plot, hist_b_plot = compute_masked_hsv_histogram_distance(patch_a, mask_a, patch_b, mask_b)
    
    print(f"--- {title_tag} Diagnostics ---")
    print(f"Bhattacharyya Distance: {distance:.4f}\n")
    
    # Mask out everything except the specific character for visual tracking
    visual_a = cv2.bitwise_and(patch_a, patch_a, mask=mask_a)
    visual_b = cv2.bitwise_and(patch_b, patch_b, mask=mask_b)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{title_tag} Analysis Dashboard\nBhattacharyya Distance: {distance:.4f}", 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Character Visualizations
    axes[0, 0].imshow(cv2.cvtColor(visual_a, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(label_a)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(visual_b, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(label_b)
    axes[0, 1].axis('off')
    
    # Row 2: Histograms
    im1 = axes[1, 0].imshow(hist_a_plot, interpolation='nearest', origin='lower', extent=[0, 256, 0, 180], cmap='viridis')
    axes[1, 0].set_title(f"{label_a} H-S Profile")
    axes[1, 0].set_xlabel("Saturation (S)")
    axes[1, 0].set_ylabel("Hue (H)")
    fig.colorbar(im1, ax=axes[1, 0], label='Log Distribution Intensity')
    
    im2 = axes[1, 1].imshow(hist_b_plot, interpolation='nearest', origin='lower', extent=[0, 256, 0, 180], cmap='viridis')
    axes[1, 1].set_title(f"{label_b} H-S Profile")
    axes[1, 1].set_xlabel("Saturation (S)")
    axes[1, 1].set_ylabel("Hue (H)")
    fig.colorbar(im2, ax=axes[1, 1], label='Log Distribution Intensity')
    
    plt.tight_layout()
    plt.show()
    
    return distance


if __name__ == "__main__":
    # 1. Paths Setup
    img_template_path = r'/home/yuval-rubin/Projects/tennis_ball/images/template_new_ball.jpg'
    img_runtime_path = r'/home/yuval-rubin/Projects/tennis_ball/images/deform_L_N_Symbol.jpg'
    config_path = r'/home/yuval-rubin/Projects/tennis_ball/balls_config/tretorn_serie_plus_control.json'

    # 2. Process images
    processor = Processor(config_path)
    outputs_template = processor(img_template_path)
    outputs_runtime = processor(img_runtime_path)
    
    if outputs_template is not None and outputs_runtime is not None:
        # 3. Unpack ONLY the components we care about (cropped ball and character lists)
        tpl_ball, _, _, _, tpl_chars, _ = outputs_template
        run_ball, _, _, _, run_chars, _ = outputs_runtime
        
        # 4. Global spatial alignment check for the base images
        if run_ball.shape != tpl_ball.shape:
            target_size = (tpl_ball.shape[1], tpl_ball.shape[0])
            run_ball = cv2.resize(run_ball, target_size, interpolation=cv2.INTER_LINEAR)
            
        print("====================================================")
        print(" STARTING LOGO CHARACTER-BY-CHARACTER COMPARISON    ")
        print("====================================================\n")
        
        print(f"Detected Template Characters: {len(tpl_chars)}")
        print(f"Detected Runtime Characters: {len(run_chars)}")
        
        num_comparisons = min(len(tpl_chars), len(run_chars))
        char_distances = []
        
        if num_comparisons == 0:
            print("Warning: One of the images has 0 segmented characters. Loop cancelled.")
        else:
            # 5. Core execution loop exclusively comparing char vs char
            for idx in range(num_comparisons):
                # FIX: Access the dictionary item using the 'mask' key
                tpl_char_mask = tpl_chars[idx]['mask']
                run_char_mask = run_chars[idx]['mask']
                
                print(f"--> Analyzing Character Pair #{idx + 1}...")
                dist_char = evaluate_and_plot_character_comparison(
                    patch_a=tpl_ball, mask_a=tpl_char_mask,
                    patch_b=run_ball, mask_b=run_char_mask,
                    label_a=f"Template Char #{idx + 1}", label_b=f"Runtime Char #{idx + 1}",
                    title_tag=f"Character {idx + 1} Comparison"
                )
                char_distances.append(dist_char)
        
            # 6. Final Metric Summary Report
            print("\n====================================================")
            print(" CHARACTER ANALYSIS RESULTS ")
            print("====================================================")
            for i, dist in enumerate(char_distances):
                print(f"Character Pair #{i + 1} Color Distance: {dist:.4f}")
            print(f"Overall Mean Character Degradation: {np.mean(char_distances):.4f}")
            print("====================================================")

    else:
        print("Error: Processing failed on one or both images.")