"""
Advanced Dartboard Segment Detector

This module implements multiple approaches for dartboard detection and segmentation:
1. Colour-based detection (fastest, works with distinct colours)
2. Contour-based detection (robust to lighting)
3. Radial line detection (accurate segment boundaries)
4. Template-based scoring (uses known dartboard layout)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import math

class DartboardDetector:
    """
    Comprehensive dartboard detector using multiple computer vision techniques.
    
    Approach:
    1. Find the dartboard using colour + contour detection
    2. Locate centre and normalise perspective
    3. Detect radial boundaries using edge detection
    4. Map segments to known score layout
    """
    
    # Standard dartboard configuration
    # Score 20 is at the top (12 o'clock position) and scores proceed clockwise
    SEGMENT_SCORES = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    NUM_SEGMENTS = 20
    SEGMENT_ANGLE = 360 / NUM_SEGMENTS  # 18 degrees per segment
    
    # Ring ratios (relative to dartboard radius - the outer double ring)
    # Standard dartboard dimensions:
    # - Outer double ring: 170mm radius (reference = 1.00)
    # - Inner double ring: 162mm radius
    # - Outer triple ring: 107mm radius
    # - Inner triple ring: 99mm radius
    # - Outer bull: 15.9mm radius
    # - Inner bull: 6.35mm radius
    RATIO_OUTER_DOUBLE = 1.00   # 170mm / 170mm - This IS the outer radius
    RATIO_INNER_DOUBLE = 0.953  # 162mm / 170mm
    RATIO_OUTER_TRIPLE = 0.629  # 107mm / 170mm
    RATIO_INNER_TRIPLE = 0.582  # 99mm / 170mm
    RATIO_OUTER_BULL = 0.094    # 15.9mm / 170mm
    RATIO_INNER_BULL = 0.037    # 6.35mm / 170mm
    
    def __init__(self, debug=True):
        """Initialise the detector."""
        self.debug = debug
        logging.basicConfig(
            level=logging.INFO if debug else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_dartboard(self, image: np.ndarray) -> Optional[Dict]:
        """
        Main detection method that finds the dartboard using multiple techniques.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with dartboard properties or None if not found
        """
        self.logger.info("Starting dartboard detection")
        
        # Try colour-based detection (works great with orange surround)
        result = self._detect_by_colour(image)
        if result:
            self.logger.info("✓ Dartboard found using colour detection")
            return result
            
        # Try contour-based detection
        result = self._detect_by_contour(image)
        if result:
            self.logger.info("✓ Dartboard found using contour detection")
            return result
            
        # Try circle detection (Hough)
        result = self._detect_by_circles(image)
        if result:
            self.logger.info("✓ Dartboard found using circle detection")
            return result
            
        self.logger.warning("✗ No dartboard detected")
        return None
    
    def _detect_by_colour(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect dartboard using colour segmentation (orange surround).
        This is the fastest and most reliable method for boards with coloured surrounds.
        """
        self.logger.info("Attempting colour-based detection")
        
        # Convert to HSV for better colour segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define orange colour range (for Winmau surround)
        lower_orange1 = np.array([0, 100, 100])
        upper_orange1 = np.array([15, 255, 255])
        lower_orange2 = np.array([160, 100, 100])
        upper_orange2 = np.array([180, 255, 255])
        
        # Create and combine masks
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find best circular contour
        best_contour = self._find_best_circular_contour(mask)
        
        if best_contour is not None:
            cx, cy, outer_radius = self._get_contour_centre_and_radius(best_contour)
            
            # Refine centre by finding the bullseye
            refined_centre = self._refine_centre_with_bullseye(image, (cx, cy), outer_radius)
            if refined_centre:
                cx, cy = refined_centre
                self.logger.info("Refined centre using bullseye detection")
            
            # The orange surround is larger than the actual dartboard
            # Dartboard playing surface is approximately 49% of surround radius
            dartboard_radius = int(outer_radius * 0.49)
            
            return {
                'centre': (cx, cy),
                'radius': dartboard_radius,
                'outer_radius': outer_radius,
                'method': 'colour',
                'mask': mask
            }
        
        return None
    
    def _find_best_circular_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the best circular contour in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Too small
                continue
                
            # Calculate circularity
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = math.pi * radius * radius
            circularity = area / circle_area if circle_area > 0 else 0
            
            # Score based on size and circularity
            score = area * circularity
            
            if circularity > 0.6 and score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def _get_contour_centre_and_radius(self, contour: np.ndarray) -> Tuple[int, int, int]:
        """Get centre coordinates and radius from a contour."""
        # Use moments for more accurate centre calculation
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to enclosing circle centre
            (cx, cy), _ = cv2.minEnclosingCircle(contour)
            cx, cy = int(cx), int(cy)
        
        # Get radius from enclosing circle
        _, radius = cv2.minEnclosingCircle(contour)
        return cx, cy, int(radius)
    
    def _refine_centre_with_bullseye(self, image: np.ndarray, 
                                     approx_centre: Tuple[int, int], 
                                     outer_radius: int) -> Optional[Tuple[int, int]]:
        """
        Refine the centre by detecting the actual bullseye with high precision.
        
        Uses multiple techniques:
        1. Colour-based detection (red/green bullseye colours)
        2. Circle detection with Hough transform
        
        Args:
            image: Input image
            approx_centre: Approximate centre from surround detection
            outer_radius: Outer radius of the surround
            
        Returns:
            Refined centre coordinates or None if bullseye not found
        """
        self.logger.info("Refining centre using bullseye detection")
        
        # Create search region
        search_radius = int(outer_radius * 0.12)
        x, y = approx_centre
        x_min = max(0, x - search_radius)
        x_max = min(image.shape[1], x + search_radius)
        y_min = max(0, y - search_radius)
        y_max = min(image.shape[0], y + search_radius)
        
        search_region = image[y_min:y_max, x_min:x_max]
        if search_region.size == 0:
            return None
        
        # Method 1: Look for red/green colours in the bullseye
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Red bullseye centre (inner bull)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Green outer bull
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # Combine masks
        bullseye_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bullseye_mask = cv2.morphologyEx(bullseye_mask, cv2.MORPH_CLOSE, kernel)
        bullseye_mask = cv2.morphologyEx(bullseye_mask, cv2.MORPH_OPEN, kernel)
        
        # Find the best circular contour in the bullseye mask
        refined_centre = self._find_centre_from_mask(bullseye_mask, search_region)
        if refined_centre:
            refined_x = int(x_min + refined_centre[0])
            refined_y = int(y_min + refined_centre[1])
            offset_x = refined_x - approx_centre[0]
            offset_y = refined_y - approx_centre[1]
            self.logger.info(f"Refined centre by ({offset_x:+d}, {offset_y:+d}) pixels")
            return (refined_x, refined_y)
        
        # Method 2: Fallback to Hough circles
        self.logger.info("Colour-based detection failed, trying Hough circles")
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
            param1=50, param2=15, minRadius=3,
            maxRadius=int(outer_radius * 0.10)
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Find the circle closest to the centre of search region
            centre_of_search = (search_region.shape[1] / 2, search_region.shape[0] / 2)
            best_circle = min(circles[0], 
                            key=lambda c: math.sqrt((c[0] - centre_of_search[0])**2 + 
                                                   (c[1] - centre_of_search[1])**2))
            
            refined_x = int(x_min + best_circle[0])
            refined_y = int(y_min + best_circle[1])
            return (refined_x, refined_y)
        
        return None
    
    def _find_centre_from_mask(self, mask: np.ndarray, reference_image: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find the centre of the most suitable contour in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centre_of_search = (reference_image.shape[1] / 2, reference_image.shape[0] / 2)
        best_centre = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Too small
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # Reasonably circular
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    
                    # Distance from approximate centre
                    dist = math.sqrt((cx - centre_of_search[0])**2 + (cy - centre_of_search[1])**2)
                    
                    # Score: prefer circular shapes near the centre
                    score = circularity * (1.0 / (1.0 + dist * 0.1))
                    
                    if score > best_score:
                        best_score = score
                        best_centre = (cx, cy)
        
        return best_centre
    
    def _detect_by_contour(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect dartboard by finding the most circular contour.
        Works well when colour detection fails.
        """
        self.logger.info("Attempting contour-based detection")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find best circular contour
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        best_circle = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Minimum area
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > best_circularity and circularity > 0.7:
                best_circularity = circularity
                (x, y), radius = cv2.minEnclosingCircle(contour)
                best_circle = (int(x), int(y), int(radius))
        
        if best_circle:
            return {
                'centre': (best_circle[0], best_circle[1]),
                'radius': best_circle[2],
                'method': 'contour',
                'circularity': best_circularity
            }
        
        return None
    
    def _detect_by_circles(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect dartboard using Hough Circle Transform.
        Most computationally expensive but can work in difficult conditions.
        """
        self.logger.info("Attempting circle detection (Hough)")
        
        # Resize for faster processing
        scale = 0.5
        small = cv2.resize(image, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=50, maxRadius=300
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Take the largest circle
            largest = max(circles[0], key=lambda c: c[2])
            x, y, r = largest
            x, y, r = int(x / scale), int(y / scale), int(r / scale)
            
            return {
                'centre': (x, y),
                'radius': r,
                'method': 'hough'
            }
        
        return None
    
    def detect_segments(self, image: np.ndarray, dartboard_info: Dict) -> Dict:
        """
        Detect the 20 segments of the dartboard.
        Currently uses uniform distribution with rotation detection.
        
        Args:
            image: Input image
            dartboard_info: Dictionary with centre and radius
            
        Returns:
            Dictionary with segment information
        """
        self.logger.info("Detecting dartboard segments")
        
        centre = dartboard_info['centre']
        radius = dartboard_info['radius']
        
        # Use uniform distribution (18° spacing)
        segment_angles = [i * self.SEGMENT_ANGLE for i in range(self.NUM_SEGMENTS)]
        
        return {
            'centre': centre,
            'radius': radius,
            'segment_angles': segment_angles,
            'num_segments': self.NUM_SEGMENTS
        }
    
    def visualise_segments(self, image: np.ndarray, segment_info: Dict, 
                          output_path: str = None, manual_rotation: Optional[float] = None) -> np.ndarray:
        """
        Create a visualisation with all segments highlighted and labelled.
        
        Args:
            image: Input image
            segment_info: Segment detection results
            output_path: Optional path to save visualisation
            manual_rotation: Optional manual rotation offset in degrees for testing
        """
        self.logger.info("Creating segment visualisation")
        
        result = image.copy()
        centre = segment_info['centre']
        radius = segment_info['radius']
        
        # Draw scoring rings
        ring_info = [
            (self.RATIO_OUTER_DOUBLE, (0, 200, 0), 2, "Double outer"),
            (self.RATIO_INNER_DOUBLE, (0, 200, 0), 2, "Double inner"),
            (self.RATIO_OUTER_TRIPLE, (200, 0, 200), 2, "Triple outer"),
            (self.RATIO_INNER_TRIPLE, (200, 0, 200), 2, "Triple inner"),
            (self.RATIO_OUTER_BULL, (255, 255, 0), 2, "Outer bull"),
            (self.RATIO_INNER_BULL, (0, 0, 255), 2, "Inner bull"),
        ]
        
        for ratio, colour, thickness, label in ring_info:
            r = int(radius * ratio)
            cv2.circle(result, centre, r, colour, thickness)
        
        # Detect or use manual rotation
        if manual_rotation is not None:
            rotation_offset = manual_rotation
            self.logger.info(f"Using MANUAL rotation offset: {rotation_offset:.2f}°")
        else:
            rotation_offset = self._detect_rotation_offset(image, centre, radius)
            self.logger.info(f"Using detected rotation offset: {rotation_offset:.2f}°")
        
        # Standard dartboard layout: 20 is at TOP (270° in image coordinates)
        # Segments go CLOCKWISE: 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
        segment_boundary_start = -90 - (self.SEGMENT_ANGLE / 2) + rotation_offset
        
        # Draw radial lines for segment boundaries
        for i in range(self.NUM_SEGMENTS):
            angle = segment_boundary_start + (i * self.SEGMENT_ANGLE)
            rad = math.radians(angle)
            x_end = int(centre[0] + radius * 1.05 * math.cos(rad))
            y_end = int(centre[1] + radius * 1.05 * math.sin(rad))
            cv2.line(result, centre, (x_end, y_end), (255, 255, 0), 2)
        
        # Add score labels at segment centres
        for i in range(self.NUM_SEGMENTS):
            score = self.SEGMENT_SCORES[i]
            angle = segment_boundary_start + (i * self.SEGMENT_ANGLE) + (self.SEGMENT_ANGLE / 2)
            rad = math.radians(angle)
            
            # Position label outside the board
            label_radius = int(radius * 1.15)
            x = int(centre[0] + label_radius * math.cos(rad))
            y = int(centre[1] + label_radius * math.sin(rad))
            
            # Draw score with background for visibility
            text = str(score)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Background rectangle
            padding = 5
            cv2.rectangle(result, 
                         (x - text_width//2 - padding, y - text_height//2 - padding),
                         (x + text_width//2 + padding, y + text_height//2 + padding),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result, text, (x - text_width//2, y + text_height//2),
                       font, font_scale, (255, 255, 0), thickness)
        
        # Draw centre point
        cv2.circle(result, centre, 8, (0, 0, 255), -1)
        cv2.circle(result, centre, 10, (255, 255, 255), 2)
        
        # Add legend
        self._draw_legend(result, centre, radius, segment_info)
        
        if output_path:
            cv2.imwrite(output_path, result)
            self.logger.info(f"Saved visualisation to {output_path}")
        
        return result
    
    def _draw_legend(self, image: np.ndarray, centre: Tuple[int, int], 
                    radius: int, segment_info: Dict):
        """Draw an informative legend on the visualisation."""
        legend_y = 30
        legend_x = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background for legend
        cv2.rectangle(image, (5, 5), (450, 135), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (450, 135), (255, 255, 255), 2)
        
        cv2.putText(image, "Dartboard Segment Detection", (legend_x, legend_y),
                   font, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Centre: {centre}", (legend_x, legend_y + 30),
                   font, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Radius: {radius}px", (legend_x, legend_y + 55),
                   font, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Method: {segment_info.get('method', 'N/A')}", (legend_x, legend_y + 80),
                   font, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f"Segments: 20 (uniform)", (legend_x, legend_y + 105),
                   font, 0.5, (200, 200, 200), 1)
    
    def _detect_rotation_offset(self, image: np.ndarray, centre: Tuple[int, int], 
                               radius: int) -> float:
        """
        Detect the rotation offset of the dartboard by analysing wire positions.
        
        Returns:
            Rotation offset in degrees (0 = perfect alignment, positive = clockwise rotation)
        """
        self.logger.info("Detecting board rotation offset")
        
        # Sample points along multiple radii
        sample_radii = [int(radius * r) for r in [0.55, 0.65, 0.75]]
        num_samples = 720  # 0.5 degree precision
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Sample edge intensity at each angle
        angular_profile = []
        for i in range(num_samples):
            angle_deg = (i * 360.0 / num_samples)
            angle_rad = math.radians(angle_deg)
            
            intensity_sum = 0
            count = 0
            
            for sample_radius in sample_radii:
                x = int(centre[0] + sample_radius * math.cos(angle_rad))
                y = int(centre[1] + sample_radius * math.sin(angle_rad))
                
                if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                    sample_region = edges[max(0, y-3):min(edges.shape[0], y+4), 
                                         max(0, x-3):min(edges.shape[1], x+4)]
                    if sample_region.size > 0:
                        intensity_sum += np.mean(sample_region)
                        count += 1
            
            avg_intensity = intensity_sum / count if count > 0 else 0
            angular_profile.append(avg_intensity)
        
        # Smooth the profile
        kernel_size = 11
        smoothed = np.convolve(angular_profile, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find peaks (wire locations)
        try:
            from scipy import signal
            
            min_distance = int(15 * num_samples / 360)
            peaks, properties = signal.find_peaks(
                smoothed, 
                distance=min_distance,
                prominence=np.std(smoothed) * 0.4,
                height=np.mean(smoothed) * 0.5
            )
            
            if len(peaks) < 15 or len(peaks) > 25:
                self.logger.warning(f"Detected {len(peaks)} wires (expected ~20), using default")
                return 0.0
            
            # Calculate median offset from expected positions
            peak_angles = [(p * 360.0 / num_samples) for p in peaks]
            expected_boundaries = [(261 - i * 18) % 360 for i in range(20)]
            
            offsets = []
            for peak_angle in peak_angles:
                closest_expected = min(expected_boundaries, 
                                      key=lambda e: min(abs(peak_angle - e), 360 - abs(peak_angle - e)))
                offset = peak_angle - closest_expected
                if offset > 180:
                    offset -= 360
                if offset < -180:
                    offset += 360
                offsets.append(offset)
            
            median_offset = np.median(offsets)
            self.logger.info(f"Detected {len(peaks)} wires, rotation offset: {median_offset:.1f}°")
            return median_offset
            
        except ImportError:
            self.logger.warning("scipy not available, using default alignment")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error in rotation detection: {e}, using default")
            return 0.0
    
    def get_segment_score(self, point: Tuple[int, int], segment_info: Dict) -> Dict:
        """
        Get the score for a point on the dartboard.
        
        Args:
            point: (x, y) coordinates
            segment_info: Segment detection results
            
        Returns:
            Dictionary with score information
        """
        centre = segment_info['centre']
        radius = segment_info['radius']
        
        # Calculate distance from centre
        dx = point[0] - centre[0]
        dy = point[1] - centre[1]
        distance = math.sqrt(dx*dx + dy*dy)
        distance_ratio = distance / radius
        
        # Determine ring
        if distance_ratio <= self.RATIO_INNER_BULL:
            return {'score': 50, 'multiplier': 1, 'region': 'inner_bull', 'total': 50}
        elif distance_ratio <= self.RATIO_OUTER_BULL:
            return {'score': 25, 'multiplier': 1, 'region': 'outer_bull', 'total': 25}
        
        # Calculate angle
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        
        # Calculate segment index
        segment_boundary_start = 261  # -99° + 360° = 261°
        adjusted_angle = (angle - segment_boundary_start) % 360
        segment_idx = int(adjusted_angle / self.SEGMENT_ANGLE) % self.NUM_SEGMENTS
        base_score = self.SEGMENT_SCORES[segment_idx]
        
        # Determine multiplier based on ring
        if self.RATIO_INNER_TRIPLE <= distance_ratio <= self.RATIO_OUTER_TRIPLE:
            multiplier = 3
            region = 'triple'
        elif self.RATIO_INNER_DOUBLE <= distance_ratio <= self.RATIO_OUTER_DOUBLE:
            multiplier = 2
            region = 'double'
        else:
            multiplier = 1
            region = 'single'
        
        return {
            'score': base_score,
            'multiplier': multiplier,
            'region': region,
            'total': base_score * multiplier,
            'segment': segment_idx,
            'angle': angle,
            'distance': distance
        }
