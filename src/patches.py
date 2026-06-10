import cv2
import numpy as np
import matplotlib.pyplot as plt
from processor import Processor

def compute_hsv_histogram_distance(patch_a, patch_b):
    """
    Computes the Bhattacharyya distance between the 2D H-S histograms 
    of two patches. Returns 0.0 for a perfect match, 1.0 for complete mismatch.
    """
    hsv_a = cv2.cvtColor(patch_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(patch_b, cv2.COLOR_BGR2HSV)
    
    # 2D Histogram over Hue and Saturation channels
    # Using 128 bins for Hue, 128 bins for Saturation as specified in your setup
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [128, 128], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [128, 128], [0, 180, 0, 256])
    
    # Store original histograms for plotting before normalization
    hist_a_plot = np.log1p(hist_a)
    hist_b_plot = np.log1p(hist_b)
    
    # Normalize histograms for the distance formula match
    cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    
    return distance, hist_a_plot, hist_b_plot


def evaluate_and_plot_comparison(runtime_patch, template_gt_patch, label="Lower Patch"):
    """
    Evaluates patch similarity using only the Color Axis and displays
    the image crops alongside their 2D H-S histograms side by side.
    """
    color_distance, hist_temp, hist_runt = compute_hsv_histogram_distance(template_gt_patch, runtime_patch)
    
    print(f"--- {label} Comparison Diagnostics ---")
    print(f"Color Distance (Bhattacharyya): {color_distance:.4f}")
    
    # Setup 2x2 plotting dashboard
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{label} Color Analysis Dashboard\nBhattacharyya Distance: {color_distance:.4f}", 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Patch Visuals (Convert BGR to RGB for Matplotlib)
    axes[0, 0].imshow(cv2.cvtColor(template_gt_patch, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Template {label}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(runtime_patch, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Runtime {label}")
    axes[0, 1].axis('off')
    
    # Row 2: Heatmap H-S Histograms
    # Template Hist
    im1 = axes[1, 0].imshow(hist_temp, interpolation='nearest', origin='lower', 
                             extent=[0, 256, 0, 180], cmap='viridis')
    axes[1, 0].set_title("Template 2D H-S Profile")
    axes[1, 0].set_xlabel("Saturation (S)")
    axes[1, 0].set_ylabel("Hue (H)")
    fig.colorbar(im1, ax=axes[1, 0], label='Log Distribution Intensity')
    
    # Runtime Hist
    im2 = axes[1, 1].imshow(hist_runt, interpolation='nearest', origin='lower', 
                             extent=[0, 256, 0, 180], cmap='viridis')
    axes[1, 1].set_title("Runtime 2D H-S Profile")
    axes[1, 1].set_xlabel("Saturation (S)")
    axes[1, 1].set_ylabel("Hue (H)")
    fig.colorbar(im2, ax=axes[1, 1], label='Log Distribution Intensity')
    
    plt.tight_layout()
    plt.show()
    
    return color_distance


# --- Execution Example ---
if __name__ == "__main__":
    # 1. Define paths
    img_template_path = r'/home/yuval-rubin/Projects/tennis_ball/images/template_new_ball.jpg'
    img_runtime_path = r'/home/yuval-rubin/Projects/tennis_ball/images/deform_R.jpg'
    config_path = r'/home/yuval-rubin/Projects/tennis_ball/balls_config/tretorn_serie_plus_control.json'

    # 2. Initialize the processor
    processor = Processor(config_path)
    
    # 3. Process both images to extract their respective patches
    outputs_template = processor(img_template_path)
    outputs_runtime = processor(img_runtime_path)
    
    if outputs_template is not None and outputs_runtime is not None:
        # 4. Unpack only the lower patches (index 5)
        # Replacing unused assignments with underscores to save memory
        _, _, _, _, _, template_lower_patch = outputs_template
        _, _, _, _, _, runtime_lower_patch = outputs_runtime
        
        # Check size alignment and resize runtime if needed to match template
        if runtime_lower_patch.shape != template_lower_patch.shape:
            runtime_lower_patch = cv2.resize(
                runtime_lower_patch, 
                (template_lower_patch.shape[1], template_lower_patch.shape[0])
            )
            
        # 5. Evaluate and plot the Lower Patch Color Distribution
        lower_distance = evaluate_and_plot_comparison(runtime_lower_patch, template_lower_patch, label="Lower Patch")
        
        print(f"\n=====================================")
        print(f"Overall Ball Color Distance: {lower_distance:.4f}")
        print(f"=====================================")
        
    else:
        print("Error: One or both of the ball images failed processing stages.")