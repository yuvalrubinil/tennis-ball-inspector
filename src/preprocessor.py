import cv2
import numpy as np
import json


class Preprocessor:

    def __init__(self, config_path):
        with open(config_path, 'r') as config:
            self.config = json.load(config)

    def detect_circle(self, image):
        """
        Runs Hough circle algorithem to detect the tennis ball circle.
        Algorithem: https://www.youtube.com/watch?v=Ltqt24SQQoI 
        """
        min_radius, max_radius, _ = self.config["circle"].values()


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
            minRadius=min_radius, 
            maxRadius=max_radius
        )
        
        if detected_circles is not None:
            # convert coordinates to integers
            detected_circles = np.uint16(np.around(detected_circles))

            # highest accumulator score
            clearest_circle = detected_circles[0, 0]
            a, b, r =  clearest_circle
            return int(a), int(b), int(r)
        
        return None 


    def crop_ball(self, image, center):
        """
        Crops a square of the image around the center of the ball.
        """
        _, _, crop_size = self.config["circle"].values()

        x, y = center
        half_size = crop_size // 2
        img_h, img_w = image.shape[:2]
        
        # new coordinates
        y_min = y - half_size
        y_max = y + half_size
        x_min = x - half_size
        x_max = x + half_size

        # in case of out of bounds:
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - img_h)
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - img_w)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            y_min += pad_top
            y_max += pad_top
            x_min += pad_left
            x_max += pad_left


        # slicing the image
        cropped_ball = image[y_min:y_max, x_min:x_max]
        
        return cropped_ball
    

    def seam_lines_roi(self, cropped_image):
        """
        Returns a bin map of the seam lines ROI.
        """
        img_h, img_w = cropped_image.shape[:2]
        u_up, u_low, u_left, u_right = self.config["upper_line"].values()
        upper_line_upper_limit = int(img_h * u_up)
        upper_line_lower_limit = int(img_h * u_low)
        upper_line_left_limit  = int(img_w * u_left)
        upper_line_right_limit = int(img_w * u_right)

        l_up, l_low, l_left, l_right = self.config["lower_line"].values()
        lower_line_upper_limit = int(img_h * l_up)
        lower_line_lower_limit = int(img_h * l_low)
        lower_line_left_limit  = int(img_w * l_left)
        lower_line_right_limit = int(img_w * l_right)

        # convert to HSV to get saturation (seam lines are bright white/gray -> low saturation)
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv)
        
        # smooth the ball fuzz texture
        s_blurred = cv2.GaussianBlur(s, (7, 7), 0)
        
        # threshold saturation
        _, seam_mask = cv2.threshold(s_blurred, 110, 255, cv2.THRESH_BINARY_INV)
        
        
        upper_line_mask = np.zeros_like(seam_mask)
        lower_line_mask = np.zeros_like(seam_mask)
        
        # slicing
        upper_line_mask[upper_line_upper_limit:upper_line_lower_limit, upper_line_left_limit:upper_line_right_limit] = \
            seam_mask[upper_line_upper_limit:upper_line_lower_limit, upper_line_left_limit:upper_line_right_limit]
            
        lower_line_mask[lower_line_upper_limit:lower_line_lower_limit, lower_line_left_limit:lower_line_right_limit] = \
            seam_mask[lower_line_upper_limit:lower_line_lower_limit, lower_line_left_limit:lower_line_right_limit]
        
        # morphological open: erosion -> dilation for noise clean-up.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
        
        final_upper_seam = cv2.morphologyEx(upper_line_mask, cv2.MORPH_OPEN, horizontal_kernel)
        final_lower_seam = cv2.morphologyEx(lower_line_mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        return final_upper_seam, final_lower_seam
    

    def get_line_angle(self, seam_line_bin_mask):
        """
        Finds the angle (in degrees) of the best-fit line through the white pixels.
        """
        # coordinates of all white pixels
        coordinates = cv2.findNonZero(seam_line_bin_mask)
        if coordinates is None: return None
        coordinates = coordinates.reshape(-1, 2)
        
        # dx, dy are the normalized direction vectors
        dx, dy, _, _ = cv2.fitLine(coordinates, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # calculate angle
        angle = np.arctan2(dy, dx)[0]
        angle = np.degrees(angle)
        return angle
    

    def rotate_ball(self,cropped_image, upper_seam_line_mask, lower_seam_line_mask):
        angle_upper = self.get_line_angle(upper_seam_line_mask)
        angle_lower = self.get_line_angle(lower_seam_line_mask)
        
        if angle_upper is None and angle_lower is None:
            print("could not calculate rotation angle.")
            return cropped_image
        elif angle_upper is None:
            avg_angle = angle_lower
        elif angle_lower is None:
            avg_angle = angle_upper
        else:
            avg_angle = (angle_upper + angle_lower) / 2.0
        
        h, w = cropped_image.shape[:2]
        center = (w // 2, h // 2)
        
        # get the rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, avg_angle, scale=1.0)
        
        # rotational transformation
        straightened_image = cv2.warpAffine(
            cropped_image, 
            rot_matrix, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=[0, 0, 0]
        )

        h, w = upper_seam_line_mask.shape[:2]
        straightened_upper_seam = cv2.warpAffine(
            upper_seam_line_mask, 
            rot_matrix, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=[0, 0, 0]
        )

        h, w = lower_seam_line_mask.shape[:2]
        straightened_lower_seam = cv2.warpAffine(
            lower_seam_line_mask, 
            rot_matrix, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=[0, 0, 0]
        )
        
        return straightened_image, straightened_upper_seam, straightened_lower_seam
    

    def logo_roi(self, cropped_image):
        h, w = cropped_image.shape[:2]
        l_up, l_low, l_left, l_right = self.config["logo"].values()
        logo_upper_limit = int(h * l_up)
        logo_lower_limit = int(h * l_low)
        logo_left_limit  = int(w * l_left)
        logo_right_limit = int(w * l_right)

        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[logo_upper_limit:logo_lower_limit, logo_left_limit:logo_right_limit] = 255

        # convert to grayscale blurred
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # threshold the dark elements
        _, logo_mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        # apply only for the ROI
        logo_mask = cv2.bitwise_and(logo_mask, border_mask)
        
        # morphological close: dilation -> erosion for clearer logo letters
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))
        clean_logo_mask = cv2.morphologyEx(logo_mask, cv2.MORPH_CLOSE, vertical_kernel)
        
        return clean_logo_mask


    def segment_individual_characters(self, clean_logo_mask):
        """
        returns a list of dicts containing the mask, bbox, and centroid for each character
        """
        # connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_logo_mask)
        
        # filter out noise (area > 300) 
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 300: 
                valid_components.append({
                    'id': i,
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP],
                    'w': stats[i, cv2.CC_STAT_WIDTH],
                    'h': stats[i, cv2.CC_STAT_HEIGHT],
                    'centroid': centroids[i]
                })
        
        # sort components
        valid_components.sort(key=lambda c: c['y'])
        sorted_components = []
        current_row = []
        
        if valid_components:
            row_y = valid_components[0]['y']
            for c in valid_components:
                if c['y'] - row_y < 50:  # same row tolerance
                    current_row.append(c)
                else:
                    current_row.sort(key=lambda item: item['x'])
                    sorted_components.extend(current_row)
                    current_row = [c]
                    row_y = c['y']
            current_row.sort(key=lambda item: item['x'])
            sorted_components.extend(current_row)

        # final masks and thier metadata
        character_results = []
        for comp in sorted_components:
            char_mask = np.zeros_like(clean_logo_mask)
            char_mask[labels == comp['id']] = 255
            
            character_results.append({
                'mask': char_mask,
                'bbox': (comp['x'], comp['y'], comp['w'], comp['h']),
                'centroid': comp['centroid']
            })
            
        return character_results


    def patch_roi(self, cropped_image):
        """
        Crops upper and lower patches from an image.
        """
        h, w = cropped_image.shape[:2]
        
        p_up, p_low, p_left, p_right = self.config["patch"].values()
        lower_patch_upper_limit = int(p_up * h)
        lower_patch_lower_limit = int(p_low * h)
        lower_patch_left_limit = int(p_left * w)
        lower_patch_right_limit = int(p_right * w)
        
        # slicing
        patch = cropped_image[lower_patch_upper_limit:lower_patch_lower_limit, lower_patch_left_limit:lower_patch_right_limit]
        
        return patch


    def __call__(self, img_path):
        img = cv2.imread(img_path)
        # focus on the ball
        a, b, r = self.detect_circle(img)
        cropped_ball = self.crop_ball(img, center=(a, b))

        # rotate horizontally
        upper_seam_mask, lower_seam_mask = self.seam_lines_roi(cropped_ball)
        cropped_ball, upper_seam_mask, lower_seam_mask = self.rotate_ball(cropped_ball, upper_seam_mask, lower_seam_mask)

        # get roi's
        logo_mask = self.logo_roi(cropped_ball)
        characters = self.segment_individual_characters(logo_mask)
        patch = self.patch_roi(cropped_ball)

        return cropped_ball, r, upper_seam_mask, lower_seam_mask, characters, patch


