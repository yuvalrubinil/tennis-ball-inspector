import cv2
import numpy as np
import matplotlib.pyplot as plt

#returns a, b, r
def detect_circle(image):
    """
    Runs Hough circle algorithem to detect the tennis ball circle.
    Algorithem: https://www.youtube.com/watch?v=Ltqt24SQQoI 
    """
    # convert to grayscale blurred
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_blurred = cv2.GaussianBlur(image_gray, (9, 9), 2)
    
    detected_circles = cv2.HoughCircles(
        image_gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=2, 
        minDist=1000, 
        param1=50, 
        param2=40, 
        minRadius=400, 
        maxRadius=700
    )
    
    if detected_circles is not None:
        # convert coordinates to integers
        detected_circles = np.uint16(np.around(detected_circles))

        # highest accumulator score
        clearest_circle = detected_circles[0, 0]
        a, b, r =  clearest_circle
        return int(a), int(b), int(r)
    
    return None 

# reteurns an image
def crop_centered_square(image, center, size=1500):
    """
    Crops a square of a given size centered at (x, y).
    """
    x, y = center
    half_size = size // 2
    
    img_h, img_w = image.shape[:2]
    
    # calculate new coordinates
    y_min = y - half_size
    y_max = y + half_size
    x_min = x - half_size
    x_max = x + half_size

    # In case of out of bounds:
    pad_top = max(0, -y_min)
    pad_bottom = max(0, y_max - img_h)
    pad_left = max(0, -x_min)
    pad_right = max(0, x_max - img_w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        y_min += pad_top
        y_max += pad_top
        x_min += pad_left
        x_max += pad_left

    # slice the image
    cropped_square = image[y_min:y_max, x_min:x_max]
    
    return cropped_square


# returns a bin mask
def extract_seams_by_focus_zones(cropped_image):
    # 1. Convert to HSV and isolate the Saturation channel
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    
    # 2. Smooth to blend the tennis ball fuzz texture
    s_blurred = cv2.GaussianBlur(s, (7, 7), 0)
    
    # 3. Threshold Saturation (Low saturation seams become white)
    _, seam_mask = cv2.threshold(s_blurred, 110, 255, cv2.THRESH_BINARY_INV)
    
    # Get image dimensions (1400x1400)
    img_h, img_w = seam_mask.shape[:2]
    
    # 4. Define the Focus Zones (Slices)
    upper_line_upper_limit = int(img_h * 0.1)
    upper_line_lower_limit = int(img_h * 0.25)
    upper_line_left_limit  = int(img_w * 0.35)  # Multiplied by img_w for width consistency
    upper_line_right_limit = int(img_w * 0.65)  # Multiplied by img_w for width consistency
    
    lower_line_upper_limit = int(img_h * 0.75)
    lower_line_lower_limit = int(img_h * 0.9)
    lower_line_left_limit  = int(img_w * 0.35)  # Multiplied by img_w for width consistency
    lower_line_right_limit = int(img_w * 0.65)  # Multiplied by img_w for width consistency
    
    # 5. Create isolated canvases for each line
    upper_line_mask = np.zeros_like(seam_mask)
    lower_line_mask = np.zeros_like(seam_mask)
    
    # FIX: Slice the EXACT same regions from seam_mask into the isolated masks
    upper_line_mask[upper_line_upper_limit:upper_line_lower_limit, upper_line_left_limit:upper_line_right_limit] = \
        seam_mask[upper_line_upper_limit:upper_line_lower_limit, upper_line_left_limit:upper_line_right_limit]
        
    lower_line_mask[lower_line_upper_limit:lower_line_lower_limit, lower_line_left_limit:lower_line_right_limit] = \
        seam_mask[lower_line_upper_limit:lower_line_lower_limit, lower_line_left_limit:lower_line_right_limit]
    
    # 6. Morphological clean-up on each mask separately
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    
    final_upper_seam = cv2.morphologyEx(upper_line_mask, cv2.MORPH_OPEN, horizontal_kernel)
    final_lower_seam = cv2.morphologyEx(lower_line_mask, cv2.MORPH_OPEN, horizontal_kernel)
    
    return final_upper_seam, final_lower_seam

# returns an angle
def get_line_angle(binary_mask):
    """
    Finds the angle (in degrees) of the best-fit line through the white pixels.
    """
    # Grab coordinates of all white pixels (returns array of [[y, x]])
    pts = cv2.findNonZero(binary_mask)
    
    if pts is None:
        return None
        
    # Extract just the coordinates
    pts = pts.reshape(-1, 2)
    
    # fitLine returns: [vx, vy, x0, y0] (vx, vy is the normalized direction vector)
    [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Calculate angle in radians, then convert to degrees
    angle = np.arctan2(vy, vx)[0]
    angle = np.degrees(angle)
    return angle

# returns an image
def rotate_horizontally(cropped_image, upper_mask, lower_mask):
    # 1. Get angles for both lines
    angle_upper = get_line_angle(upper_mask)
    angle_lower = get_line_angle(lower_mask)
    
    # Fallback checks in case one line wasn't detected properly
    if angle_upper is None and angle_lower is None:
        print("Could not calculate rotation angle.")
        return cropped_image
    elif angle_upper is None:
        avg_angle = angle_lower
    elif angle_lower is None:
        avg_angle = angle_upper
    else:
        # Average the two slopes
        avg_angle = (angle_upper + angle_lower) / 2.0
        #avg_angle = np.max(angle_lower + angle_upper)
    
    # 2. Rotate the image around its center point
    h, w = cropped_image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get the rotation matrix (we pass the raw angle; OpenCV handles the direction)
    rot_matrix = cv2.getRotationMatrix2D(center, avg_angle, scale=1.0)
    
    # Perform the actual affine transformation
    straightened_image = cv2.warpAffine(
        cropped_image, 
        rot_matrix, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=[0, 0, 0]
    )
    
    return straightened_image

# returns a bin mask
def extract_logo_mask(cropped_image):
    # 1. Convert to grayscale to evaluate pure brightness/intensity
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply a light blur to smooth out the fuzzy tennis ball texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Threshold the dark elements. 
    _, logo_mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Clean up any loose background pixels outside the ball area
    h, w = logo_mask.shape[:2]
    border_mask = np.zeros_like(logo_mask)
    border_mask[int(h*0.25):int(h*0.75), int(w*0.15):int(w*0.85)] = 255
    logo_mask = cv2.bitwise_and(logo_mask, border_mask)
    
    # 5. THE MIDDLE GROUND: Directional Morphological Close
    # We use a kernel that is narrow (5px wide) but tall (13px high).
    # This heals the vertical cracks in "C" and "O" without bleeding 
    # horizontally into the neighboring characters.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
    clean_logo_mask = cv2.morphologyEx(logo_mask, cv2.MORPH_CLOSE, vertical_kernel)
    
    return clean_logo_mask

# returns many bin masks
def segment_individual_characters(clean_logo_mask):
    """
    Finds all individual characters (letters and the plus symbol)
    and returns a list of isolated binary masks for each component.
    """
    # 1. Run connected components analysis with structural stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_logo_mask)
    
    # Pack the component data into a list of dictionaries for easier handling
    # Skip index 0 because it always represents the black background matrix
    components = []
    for i in range(1, num_labels):
        components.append({
            'id': i,
            'x': stats[i, cv2.CC_STAT_LEFT],
            'y': stats[i, cv2.CC_STAT_TOP],
            'w': stats[i, cv2.CC_STAT_WIDTH],
            'h': stats[i, cv2.CC_STAT_HEIGHT],
            'area': stats[i, cv2.CC_STAT_AREA],
            'centroid': centroids[i]
        })
        
    # 2. Filter out extreme noise (tiny fiber dots or stray pixels)
    # The letters in a 1400x1400 crop are large; anything under ~300 pixels is noise.
    valid_components = [c for c in components if c['area'] > 300]
    
    # 3. Sort components to read them like a book (Top-to-Bottom row-wise, then Left-to-Right)
    # We group items into rows if their Y coordinates are close (within 50 pixels)
    valid_components.sort(key=lambda c: c['y'])
    
    sorted_components = []
    current_row = []
    if valid_components:
        row_y = valid_components[0]['y']
        for c in valid_components:
            if c['y'] - row_y < 50:  # Same row tolerance
                current_row.append(c)
            else:
                # Sort the completed row left-to-right
                current_row.sort(key=lambda item: item['x'])
                sorted_components.extend(current_row)
                # Start a new row
                current_row = [c]
                row_y = c['y']
        # Don't forget the last row
        current_row.sort(key=lambda item: item['x'])
        sorted_components.extend(current_row)

    # 4. Generate isolated binary maps for each validated character
    character_masks = []
    for comp in sorted_components:
        # Create a blank canvas matching the size of the original mask
        char_mask = np.zeros_like(clean_logo_mask)
        
        # Draw ONLY this specific component's label ID onto the new mask
        char_mask[labels == comp['id']] = 255
        
        # Save both the full-frame mask and its localized properties for your defect checks
        character_masks.append({
            'mask': char_mask,
            'bbox': (comp['x'], comp['y'], comp['w'], comp['h']),
            'area': comp['area']
        })
        
    return character_masks


    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        
    letter_maps = {}
    img_h, img_w = mask_img.shape
    
    for label, (x, y, w, h) in windows.items():
        y_min, y_max = max(0, y), min(img_h, y + h)
        x_min, x_max = max(0, x), min(img_w, x + w)
        letter_maps[label] = mask_img[y_min:y_max, x_min:x_max]
        
    return letter_maps

# returns an image
def crop_patches(cropped_image):
    """
    Crops upper and lower patches from an image using hardcoded normalized limits.
    
    Args:
        cropped_image (numpy.ndarray): The input image array.
        
    Returns:
        tuple: (upper_patch, lower_patch) as numpy arrays.
    """
    # Hardcoded configuration (normalized ratios 0.0 - 1.0)
    UPPER_PATCH_UPPER_LIMIT = 0.2
    UPPER_PATCH_LOWER_LIMIT = 0.3
    UPPER_PATCH_LEFT_LIMIT = 0.4
    UPPER_PATCH_RIGHT_LIMIT = 0.6

    LOWER_PATCH_UPPER_LIMIT = 0.66
    LOWER_PATCH_LOWER_LIMIT = 0.76
    LOWER_PATCH_LEFT_LIMIT = 0.4
    LOWER_PATCH_RIGHT_LIMIT = 0.6

    # Get image dimensions (height, width)
    h, w = cropped_image.shape[:2]
    
    # Calculate pixel boundaries for the upper patch
    y1_upper = int(UPPER_PATCH_UPPER_LIMIT * h)
    y2_upper = int(UPPER_PATCH_LOWER_LIMIT * h)
    x1_upper = int(UPPER_PATCH_LEFT_LIMIT * w)
    x2_upper = int(UPPER_PATCH_RIGHT_LIMIT * w)
    
    # Calculate pixel boundaries for the lower patch
    y1_lower = int(LOWER_PATCH_UPPER_LIMIT * h)
    y2_lower = int(LOWER_PATCH_LOWER_LIMIT * h)
    x1_lower = int(LOWER_PATCH_LEFT_LIMIT * w)
    x2_lower = int(LOWER_PATCH_RIGHT_LIMIT * w)
    
    # Extract the patches using NumPy slicing [ymin:ymax, xmin:xmax]
    upper_patch = cropped_image[y1_upper:y2_upper, x1_upper:x2_upper]
    lower_patch = cropped_image[y1_lower:y2_lower, x1_lower:x2_lower]
    
    return upper_patch, lower_patch
    



# helper
def preprocess_image_to_aligned_ball(img_path):
    img = cv2.imread(img_path)
    # Executing your geometry optimization pipeline
    a, b, r = detect_circle(img)
    cropped_ball = crop_centered_square(img, center=(a, b), size=1500)
    upper_mask, lower_mask = extract_seams_by_focus_zones(cropped_ball)
    horizontal_cropped_ball = rotate_horizontally(cropped_ball, upper_mask, lower_mask)
    return horizontal_cropped_ball


paths = [
    "/home/yuval-rubin/Projects/tennis_ball/images/template_new_ball.jpg",
    "/home/yuval-rubin/Projects/tennis_ball/images/1.5h.jpg",
    "/home/yuval-rubin/Projects/tennis_ball/images/3h.jpg",
    "/home/yuval-rubin/Projects/tennis_ball/images/deform_L_N_Symbol.jpg",
    "/home/yuval-rubin/Projects/tennis_ball/images/deform_lines.jpg",
    "/home/yuval-rubin/Projects/tennis_ball/images/deform_R.jpg"
]

# Total number of images to display
num_images = len(paths)

# Setup a visualization grid: 
# Row 1: Upper Patches | Row 2: Lower Patches across all test images
fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(2.5 * num_images, 6))
fig.suptitle("Patch Inspection Matrix across All Dataset Images", fontsize=14, fontweight='bold')

# Keep axes 2D even if there's only 1 image in the list
if num_images == 1:
    axes = np.expand_dims(axes, axis=1)

for idx, img_path in enumerate(paths):
    filename = img_path.split("/")[-1]
    print(f"Extracting patches for: {filename}")
    
    # 1. Align and clean the tennis ball image
    aligned_ball = preprocess_image_to_aligned_ball(img_path)
    
    # 2. Extract upper and lower localized focus patches
    upper_p, lower_p = crop_patches(aligned_ball)
    
    # Convert from BGR (OpenCV) to RGB (Matplotlib) for accurate plotting colors
    if len(upper_p.shape) == 3:
        upper_p = cv2.cvtColor(upper_p, cv2.COLOR_BGR2RGB)
        lower_p = cv2.cvtColor(lower_p, cv2.COLOR_BGR2RGB)
    
    # --- Row 1: Upper Patch Display ---
    axes[0, idx].imshow(upper_p)
    axes[0, idx].set_title(f"Upper\n{filename}", fontsize=9)
    axes[0, idx].axis('off')
    
    # --- Row 2: Lower Patch Display ---
    axes[1, idx].imshow(lower_p)
    axes[1, idx].set_title(f"Lower\n{filename}", fontsize=9)
    axes[1, idx].axis('off')

plt.tight_layout()
plt.show()