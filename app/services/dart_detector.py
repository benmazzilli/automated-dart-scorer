"""
Simplified Dart Detector

Focused approach for detecting darts stuck in a dartboard.
Key insight: Darts have BRIGHT, SATURATED flights that stand out from the board.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import math


class DartDetector:
    """
    Simplified dart detector focused on bright, colored dart flights.
    
    Strategy:
    1. Detect bright, saturated colors (dart flights)
    2. Validate size and shape (flights are ~200-800 pixels)
    3. Look for elongated shapes projecting from board
    4. Use strict filtering to avoid false positives
    """
    
    def __init__(self, debug=True):
        """Initialise the dart detector."""
        self.debug = debug
        logging.basicConfig(
            level=logging.INFO if debug else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_darts(self, image: np.ndarray, dartboard_info: Optional[Dict] = None) -> List[Dict]:
        """
        Detect darts in the image using simplified, robust approach.
        
        Args:
            image: Input image (BGR format)
            dartboard_info: Optional dartboard information to constrain search
            
        Returns:
            List of detected darts with position, angle, and confidence
        """
        self.logger.info("Starting simplified dart detection")
        
        # Create region of interest if dartboard info provided
        roi_mask = None
        if dartboard_info:
            roi_mask = self._create_roi_mask(image, dartboard_info)
        
        # Primary method: detect bright colored flights
        darts = self._detect_colored_flights(image, roi_mask)
        
        self.logger.info(f"✓ Detected {len(darts)} darts")
        return darts
    
    def _create_roi_mask(self, image: np.ndarray, dartboard_info: Dict) -> np.ndarray:
        """
        Create ROI mask focused on dartboard area only.
        Darts stick into the board, so we search within the board radius.
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        centre = dartboard_info.get('centre')
        radius = dartboard_info.get('radius')
        
        if centre and radius:
            # Search within dartboard plus small margin
            search_radius = int(radius * 1.1)
            cv2.circle(mask, centre, search_radius, 255, -1)
            
            self.logger.info(f"ROI: centre {centre}, radius {search_radius}px")
        else:
            mask[:] = 255
        
        return mask
    
    def _detect_colored_flights(self, image: np.ndarray, roi_mask: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect darts by finding BRIGHT, SATURATED colored flights.
        This is the most reliable method as dart flights are designed to be visible.
        """
        self.logger.info("Detecting bright colored dart flights")
        
        darts = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for bright, saturated colors
        # High saturation (>120) and high value (>100) = bright colored objects
        # Balanced to catch dart flights while excluding dartboard segments
        lower = np.array([0, 120, 100])
        upper = np.array([180, 255, 255])
        saturation_mask = cv2.inRange(hsv, lower, upper)
        
        # EXCLUDE dartboard colors (red and green ranges)
        # Red range (dartboard segments)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 0]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 50, 0]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Green range (dartboard segments)
        green_mask = cv2.inRange(hsv, np.array([35, 50, 0]), np.array([85, 255, 255]))
        
        # Remove dartboard colors from detection
        dartboard_colors = cv2.bitwise_or(red_mask, green_mask)
        saturation_mask = cv2.subtract(saturation_mask, dartboard_colors)
        
        # Apply ROI mask
        if roi_mask is not None:
            saturation_mask = cv2.bitwise_and(saturation_mask, roi_mask)
        
        # Clean up: remove noise, fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_CLOSE, kernel)
        saturation_mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(saturation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.info(f"Found {len(contours)} colored regions")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Dart flights are medium-sized (200-2000 pixels depending on distance)
            # This filters out wires, segments, numbers, and tiny features
            if area < 200 or area > 2000:
                continue
            
            # Need enough points for shape analysis
            if len(contour) < 5:
                continue
            
            # Get bounding rectangle and shape properties
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect
            
            # Ensure width is the shorter dimension
            if width > height:
                width, height = height, width
                angle += 90
            
            # Dart flights are somewhat elongated (but not extremely so)
            if height > 0 and width > 0:
                aspect_ratio = height / width
            else:
                continue
            
            # Aspect ratio should be reasonable: 1.4:1 to 5:1
            # (Too round = not a dart, too elongated = wire or line)
            if aspect_ratio < 1.4 or aspect_ratio > 5.0:
                continue
            
            # Check if the contour fills the bounding box reasonably
            rect_area = width * height
            if rect_area > 0:
                solidity = area / rect_area
            else:
                continue
            
            # Dart flights should be fairly solid (not wire-like)
            if solidity < 0.4:
                continue
            
            # Calculate perimeter for additional validation
            perimeter = cv2.arcLength(contour, True)
            
            # Compactness check - reject very irregular shapes
            if perimeter > 0:
                compactness = (4 * math.pi * area) / (perimeter * perimeter)
            else:
                continue
            
            # Dart flights should have reasonable compactness (0.3-0.9)
            # Too compact = circle (bullseye), too irregular = not a dart
            if compactness < 0.2 or compactness > 0.95:
                continue
            
            # Check color intensity - dart flights are VERY BRIGHT
            mask_single = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_single, [contour], 0, 255, -1)
            mean_val = cv2.mean(hsv[:, :, 2], mask=mask_single)[0]  # V channel (brightness)
            mean_sat = cv2.mean(hsv[:, :, 1], mask=mask_single)[0]  # S channel (saturation)
            mean_hue = cv2.mean(hsv[:, :, 0], mask=mask_single)[0]  # H channel (hue)
            
            # Must be bright AND saturated (but not too strict)
            if mean_val < 100 or mean_sat < 120:
                continue
            
            # EXCLUDE dartboard red/green hues (but allow brighter versions)
            # Red: 0-15 or 165-180, Green: 35-85
            is_red = (mean_hue <= 15 or mean_hue >= 165)
            is_green = (35 <= mean_hue <= 85)
            
            # Only exclude if it's dartboard-like (dark red/green)
            if is_red and mean_val < 150 and mean_sat > 150:
                continue  # Dark red dartboard segment
            if is_green and mean_val < 130 and mean_sat > 140:
                continue  # Dark green dartboard segment
            
            # Estimate dart tip position (at one end of the flight)
            tip_x, tip_y = self._estimate_dart_tip(rect, angle, contour, image.shape)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                aspect_ratio, area, solidity, compactness, mean_sat, mean_val
            )
            
            # Only accept high-confidence detections
            if confidence < 0.65:
                continue
            
            self.logger.info(f"Dart candidate: area={area:.0f}, aspect={aspect_ratio:.2f}, "
                           f"solidity={solidity:.2f}, confidence={confidence:.2f}")
            
            darts.append({
                'position': (int(tip_x), int(tip_y)),
                'centre': (int(cx), int(cy)),
                'angle': angle,
                'confidence': confidence,
                'length': int(height),
                'width': int(width),
                'area': area,
                'method': 'colour_flight'
            })
        
        return darts
    
    def _estimate_dart_tip(self, rect: Tuple, angle: float, contour: np.ndarray, 
                          image_shape: Tuple) -> Tuple[float, float]:
        """
        Estimate dart tip position by extending from the detected flight.
        
        The steel tip is embedded in the board, several centimeters forward from
        the visible flight. We detect the flight, then project forward along
        the dart's axis to estimate the tip location.
        """
        (cx, cy), (width, height), _ = rect
        
        # Find the two endpoints of the elongated shape (flight)
        rad = math.radians(angle)
        offset = height / 2
        
        end1_x = cx + offset * math.cos(rad)
        end1_y = cy + offset * math.sin(rad)
        end2_x = cx - offset * math.cos(rad)
        end2_y = cy - offset * math.sin(rad)
        
        # Determine which end points toward the dartboard center
        # The tip is the end closer to the center
        center_x, center_y = image_shape[1] / 2, image_shape[0] / 2
        
        dist1 = math.sqrt((end1_x - center_x)**2 + (end1_y - center_y)**2)
        dist2 = math.sqrt((end2_x - center_x)**2 + (end2_y - center_y)**2)
        
        # Identify which end is the "front" (closer to center)
        if dist1 < dist2:
            front_x, front_y = end1_x, end1_y
            back_x, back_y = end2_x, end2_y
        else:
            front_x, front_y = end2_x, end2_y
            back_x, back_y = end1_x, end1_y
        
        # Calculate the dart's direction vector (from back to front)
        dx = front_x - back_x
        dy = front_y - back_y
        
        # Normalize the direction vector
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # Project forward from the front of the flight to estimate the tip position
        # A typical dart has:
        # - Flight length: ~50mm
        # - Barrel to tip: ~50-70mm
        # So the tip is roughly 1.0-1.5x the flight length forward
        # We use the detected flight length (height) to scale appropriately
        tip_offset = height * 1.2  # Project forward 1.2x the flight length
        
        tip_x = front_x + dx * tip_offset
        tip_y = front_y + dy * tip_offset
        
        return tip_x, tip_y
    
    def _calculate_confidence(self, aspect_ratio: float, area: float, 
                             solidity: float, compactness: float,
                             saturation: float, brightness: float) -> float:
        """
        Calculate confidence score based on dart flight characteristics.
        """
        # Ideal aspect ratio is around 2:1 to 3:1
        aspect_score = 1.0
        if aspect_ratio < 2.0:
            aspect_score = aspect_ratio / 2.0
        elif aspect_ratio > 3.5:
            aspect_score = 3.5 / aspect_ratio
        
        # Ideal area is 400-800 pixels
        area_score = 1.0
        if area < 400:
            area_score = area / 400
        elif area > 800:
            area_score = min(1.0, 800 / area)
        
        # Solidity should be good (0.5-0.8)
        solidity_score = min(1.0, solidity / 0.5)
        
        # Compactness should be moderate (0.3-0.7)
        compactness_score = 1.0
        if compactness < 0.3:
            compactness_score = compactness / 0.3
        elif compactness > 0.7:
            compactness_score = 0.7 / compactness
        
        # High saturation is good
        saturation_score = min(1.0, saturation / 150)
        
        # High brightness is good
        brightness_score = min(1.0, brightness / 150)
        
        # Weighted combination
        confidence = (
            aspect_score * 0.20 +
            area_score * 0.15 +
            solidity_score * 0.15 +
            compactness_score * 0.15 +
            saturation_score * 0.20 +
            brightness_score * 0.15
        )
        
        return min(1.0, max(0.0, confidence))
    
    def visualise_darts(self, image: np.ndarray, darts: List[Dict], 
                       output_path: str = None) -> np.ndarray:
        """
        Create a clean visualisation with detected darts highlighted.
        """
        self.logger.info(f"Creating visualisation for {len(darts)} darts")
        
        result = image.copy()
        
        for i, dart in enumerate(darts, 1):
            tip_pos = dart['position']
            centre_pos = dart['centre']
            angle = dart['angle']
            length = dart.get('length', 40)
            confidence = dart['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                colour = (0, 255, 0)  # Bright green - high confidence
            elif confidence > 0.7:
                colour = (0, 200, 255)  # Yellow-green
            else:
                colour = (0, 165, 255)  # Orange
            
            # Draw line from center to tip
            cv2.line(result, centre_pos, tip_pos, colour, 3)
            
            # Draw circle at flight center
            cv2.circle(result, centre_pos, 10, colour, 3)
            
            # Draw circle at dart tip
            cv2.circle(result, tip_pos, 8, colour, -1)
            cv2.circle(result, tip_pos, 10, (255, 255, 255), 2)
            
            # Add label
            label = f"Dart #{i}"
            conf_label = f"{confidence:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Position label near the dart
            label_x = tip_pos[0] - 40
            label_y = tip_pos[1] - 25
            
            # Draw background
            cv2.rectangle(result,
                         (label_x - 5, label_y - 35),
                         (label_x + 90, label_y + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result, label, (label_x, label_y - 15),
                       font, 0.5, colour, 2)
            cv2.putText(result, conf_label, (label_x, label_y),
                       font, 0.5, (255, 255, 255), 1)
        
        # Add summary
        self._draw_summary(result, darts)
        
        if output_path:
            cv2.imwrite(output_path, result)
            self.logger.info(f"Saved visualisation to {output_path}")
        
        return result
    
    def _draw_summary(self, image: np.ndarray, darts: List[Dict]):
        """Draw clean summary information."""
        summary_y = image.shape[0] - 50
        summary_x = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background
        cv2.rectangle(image, (5, image.shape[0] - 60), (280, image.shape[0] - 5), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (5, image.shape[0] - 60), (280, image.shape[0] - 5), 
                     (255, 255, 255), 2)
        
        cv2.putText(image, f"Darts detected: {len(darts)}", (summary_x, summary_y),
                   font, 0.6, (255, 255, 255), 2)
        
        if darts:
            avg_confidence = sum(d['confidence'] for d in darts) / len(darts)
            cv2.putText(image, f"Avg confidence: {avg_confidence:.2f}", 
                       (summary_x, summary_y + 25),
                       font, 0.5, (200, 200, 200), 1)
