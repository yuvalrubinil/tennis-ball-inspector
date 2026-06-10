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
    
    # Calculate 2D Histograms utilizing the specific seam line masks
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


def evaluate_and_plot_seam_comparison(patch_a, mask_a, patch_b, mask_b, label_a="Img A", label_b="Img B", title_tag="Seam"):
    """
    Isolates white mask pixels on the underlying BGR images, calculates color
    distance metrics, and plots the side-by-side comparison dashboard.
    """
    distance, hist_a_plot, hist_b_plot = compute_masked_hsv_histogram_distance(patch_a, mask_a, patch_b, mask_b)
    
    print(f"--- {title_tag} Diagnostics ---")
    print(f"Bhattacharyya Distance: {distance:.4f}\n")
    
    # Mask out everything except the seam lines for visual tracking
    visual_a = cv2.bitwise_and(patch_a, patch_a, mask=mask_a)
    visual_b = cv2.bitwise_and(patch_b, patch_b, mask=mask_b)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{title_tag} Analysis Dashboard\nBhattacharyya Distance: {distance:.4f}", 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Visualizations
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


# --- Execution Pipeline: Direct Seam-to-Seam Comparison ---
if __name__ == "__main__":
    img_template_path = r'/home/yuval-rubin/Projects/tennis_ball/images/template_new_ball.jpg'
    img_runtime_path = r'/home/yuval-rubin/Projects/tennis_ball/images/3h.jpg'
    config_path = r'/home/yuval-rubin/Projects/tennis_ball/balls_config/tretorn_serie_plus_control.json'

    processor = Processor(config_path)
    
    # Process both balls
    outputs_template = processor(img_template_path)
    outputs_runtime = processor(img_runtime_path)
    
    if outputs_template is not None and outputs_runtime is not None:
        # Unpack outputs based on your pipeline's return statement:
        # return cropped_ball, r, upper_seam_mask, lower_seam_mask, characters, lower_patch
        tpl_ball, _, tpl_up_seam, tpl_low_seam, _, _ = outputs_template
        run_ball, _, run_up_seam, run_low_seam, _, _ = outputs_runtime
        
        # Spatial alignment safety check: resize runtime dimensions to match template if they differ
        if run_ball.shape != tpl_ball.shape:
            target_size = (tpl_ball.shape[1], tpl_ball.shape[0])
            run_ball = cv2.resize(run_ball, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Using INTER_NEAREST to preserve crisp, binary 0 or 255 edges for the masks
            run_up_seam = cv2.resize(run_up_seam, target_size, interpolation=cv2.INTER_NEAREST)
            run_low_seam = cv2.resize(run_low_seam, target_size, interpolation=cv2.INTER_NEAREST)
            
        print("====================================================")
        print(" STARTING CROSS-IMAGE SEAM COMPARISON               ")
        print("====================================================\n")

        # 1. Compare Template Upper Seam vs Runtime Upper Seam
        dist_upper = evaluate_and_plot_seam_comparison(
            patch_a=tpl_ball, mask_a=tpl_up_seam,
            patch_b=run_ball, mask_b=run_up_seam,
            label_a="Template Upper Seam", label_b="Runtime Upper Seam",
            title_tag="Upper Seam Comparison"
        )
        
        # 2. Compare Template Lower Seam vs Runtime Lower Seam
        dist_lower = evaluate_and_plot_seam_comparison(
            patch_a=tpl_ball, mask_a=tpl_low_seam,
            patch_b=run_ball, mask_b=run_low_seam,
            label_a="Template Lower Seam", label_b="Runtime Lower Seam",
            title_tag="Lower Seam Comparison"
        )
        
        # Summary block
        print("====================================================")
        print(f"Final Upper Seam Distance: {dist_upper:.4f}")
        print(f"Final Lower Seam Distance: {dist_lower:.4f}")
        print("====================================================")

    else:
        print("Error: Processing failed on one or both images.")
