# Tennis Ball Inspector

![LOGO](/figures/logo1.png)

The Tennis Ball Inspector is a computer vision tool designed to assess the quality of a tennis ball by analyzing an image. It compares a given ball against a "template" image of a new, high-quality ball, evaluating various features such as size, color degradation, and the integrity of the printed logo.

## How It Works

The inspection process follows a multi-stage pipeline:

1.  **Preprocessing**: The input image is processed to isolate and standardize the tennis ball.
    *   **Ball Detection**: The ball is located using the Hough Circle Transform algorithm.
    *   **Cropping**: The image is cropped to a square region centered on the detected ball.
    *   **Orientation Correction**: The seam lines of the ball are detected by analyzing saturation levels in the HSV color space. The angle of these lines is calculated, and the cropped image is rotated so the seams are horizontal, ensuring consistent orientation for comparison.
    *   **ROI Extraction**: Specific Regions of Interest (ROIs) are identified based on the configuration file:
        *   Upper and lower seam lines.
        *   The printed logo/characters.
        *   A "patch" of the felt for general color analysis.

2.  **Feature Extraction**: Key features are extracted from both the input image and the template image.
    *   **Color Signature**: For each ROI (seams, patch, characters), a 2D color histogram is generated in the CIELAB color space, focusing on Lightness (L*) and the Blue/Yellow axis (b*). This histogram serves as a "color signature."
    *   **Structural Analysis**: The logo characters are segmented individually. Each character from the input image is aligned with its corresponding template character, and a structural difference map is created using a logical XOR operation.
    *   **Radius Measurement**: The radius of the detected ball is measured.

3.  **Comparison and Analysis**: The extracted features from the input ball are compared to the template's features.
    *   **Size**: The ratio of the input ball's radius to the template's radius is calculated.
    *   **Color Difference**: The [Earth Mover's Distance (EMD)](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) is used to quantify the difference between the color signatures of the input and template ROIs.
    *   **Structural Difference**: The structural difference map for each character is filtered to remove noise, and the remaining difference is quantified.

4.  **Reporting**: A final report is generated in the console, indicating whether each feature passes or fails based on predefined thresholds. An optional visualization dashboard can also be displayed.

## Project Structure

```
.
├── main.py                   # Main script to run an inspection
├── config/
│   └── tretorn_serie_plus_control.json # JSON file for ROI coordinates and inspection thresholds
├── images/                   # Directory for template and test images
├── src/
│   ├── ball_inspector.py     # Orchestrates the inspection and reports results
│   ├── feature_extractor.py  # Extracts and compares features against a template
│   ├── preprocessor.py       # Handles image detection, cropping, rotation, and ROI extraction
│   └── color_extractor.py    # Generates color signatures from image regions
└── figures/                  # (Optional) Directory for output figures
```

## Configuration

The inspection process is controlled by a JSON configuration file (e.g., `tretorn_serie_plus_control.json`). This file defines:

*   **ROI Coordinates**: The proportional coordinates for extracting the `circle`, `upper_line`, `lower_line`, `logo`, and `patch` from the cropped ball image.
*   **Thresholds**: The tolerance values for passing or failing the inspection. This includes:
    *   `radius_tolarence`: The acceptable min/max ratio for the ball's radius.
    *   `patch_color`: Maximum allowed color difference for the ball's felt.
    *   `seam_lines_color`: Maximum allowed color difference for the seams.
    *   `char_color`: Maximum allowed color difference for the logo characters.
    *   `char_structur`: Maximum allowed structural difference for the logo characters.

## Usage

To run an inspection, modify `main.py` with the correct file paths and execute it.

1.  **Set the paths** in `main.py`:
    ```python
    from src.ball_inspector import BallInspector

    # Path to the configuration file
    config_path = "./config/tretorn_serie_plus_control.json"

    # Path to the reference image of a new ball
    template_path = "./images/template_new_ball.jpg"

    # Path to the image of the ball you want to inspect
    image_path = "./images/deform_R.jpg"
    ```

2.  **Run the inspector**:
    ```python
    # Initialize the inspector with the config and template
    inspector = BallInspector(config_path, template_path)

    # Run the inspection on the target image
    # Set visualize_maps=True to display the dashboard
    inspector(image_path, visualize_maps=True)
    ```

3.  Execute the script from your terminal:
    ```bash
    python main.py
    ```

### Output

#### Console Report

The script will print a color-coded report to the console. Features within the acceptable tolerance are marked in green, while those that fail are marked in red.

```
Radius Ratio          : 1.0069
Patch Color Diff      : 0.003926
Seam Color Diff       : 0.008468
Valid Char Count      : True

Characters:
  Char 1: Color=0.011855, Structure=0.000302
  Char 2: Color=0.015296, Structure=0.000000
  ...
============================================================
```

#### Visualization Dashboard

If `visualize_maps=True`, a Matplotlib window will appear, showing a detailed comparison dashboard.

*   **Top Row**: Side-by-side comparison of the color signatures for the **Patch**, **Upper Seam**, and **Lower Seam** between the template and the runtime image.
*   **Middle Row**: Side-by-side comparison of the color signatures for each individual **character** in the logo.
*   **Bottom Row**: The structural difference maps for each character, highlighting areas of mismatch between the template and runtime characters. The title for each map shows the number of differing pixels after noise filtering.