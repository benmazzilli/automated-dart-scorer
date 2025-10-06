"""
Integrated Dart Scorer

Combines dartboard detection, dart detection, and scoring into a unified service.
Takes an image and produces a score with visualisation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from app.services.dartboard_detector import DartboardDetector
from app.services.dart_detector import DartDetector


class DartScorer:
    """
    Integrated dart scoring system.
    
    Takes an image and produces:
    - Total score from detected darts
    - Individual dart scores
    - Visualisation with dartboard segments and dart positions
    """
    
    def __init__(self, debug=True):
        """Initialise the dart scorer."""
        self.debug = debug
        logging.basicConfig(
            level=logging.INFO if debug else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialise detectors
        self.dartboard_detector = DartboardDetector(debug=debug)
        self.dart_detector = DartDetector(debug=debug)
    
    def score_image(self, image: np.ndarray) -> Dict:
        """
        Score an image with darts.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing:
            - total_score: Sum of all dart scores
            - dart_scores: List of individual dart score info
            - darts: List of detected darts
            - dartboard_info: Dartboard detection results
            - success: Whether scoring was successful
            - message: Status message
        """
        self.logger.info("Starting image scoring")
        
        # Step 1: Detect dartboard
        self.logger.info("Step 1: Detecting dartboard...")
        dartboard_info = self.dartboard_detector.detect_dartboard(image)
        
        if not dartboard_info:
            self.logger.warning("No dartboard detected in image")
            return {
                'total_score': 0,
                'dart_scores': [],
                'darts': [],
                'dartboard_info': None,
                'success': False,
                'message': 'No dartboard detected'
            }
        
        self.logger.info(f"✓ Dartboard found at {dartboard_info['centre']} with radius {dartboard_info['radius']}px")
        
        # Step 2: Detect segments
        self.logger.info("Step 2: Detecting segments...")
        segment_info = self.dartboard_detector.detect_segments(image, dartboard_info)
        segment_info.update(dartboard_info)
        
        # Step 3: Detect darts
        self.logger.info("Step 3: Detecting darts...")
        darts = self.dart_detector.detect_darts(image, dartboard_info)
        
        if not darts:
            self.logger.warning("No darts detected in image")
            return {
                'total_score': 0,
                'dart_scores': [],
                'darts': [],
                'dartboard_info': dartboard_info,
                'segment_info': segment_info,
                'success': True,
                'message': 'No darts detected'
            }
        
        self.logger.info(f"✓ Found {len(darts)} darts")
        
        # Step 4: Calculate scores
        self.logger.info("Step 4: Calculating scores...")
        dart_scores = []
        total_score = 0
        
        for i, dart in enumerate(darts, 1):
            score_info = self.dartboard_detector.get_segment_score(dart['position'], segment_info)
            score_info['dart_id'] = i
            score_info['position'] = dart['position']
            score_info['confidence'] = dart['confidence']
            dart_scores.append(score_info)
            total_score += score_info['total']
            
            self.logger.info(f"  Dart #{i}: {score_info['total']} points "
                           f"({score_info['score']} × {score_info['multiplier']}, "
                           f"{score_info['region']}, confidence: {dart['confidence']:.2f})")
        
        self.logger.info(f"✓ Total score: {total_score} points")
        
        return {
            'total_score': total_score,
            'dart_scores': dart_scores,
            'darts': darts,
            'dartboard_info': dartboard_info,
            'segment_info': segment_info,
            'success': True,
            'message': f'Scored {len(darts)} darts'
        }
    
    def visualise_score(self, image: np.ndarray, score_result: Dict, 
                       output_path: str = None) -> np.ndarray:
        """
        Create a clean black and white dartboard outline with dart positions as dots.
        
        Args:
            image: Input image (used only for dimensions)
            score_result: Result from score_image()
            output_path: Optional path to save visualisation
            
        Returns:
            Clean black and white dartboard diagram with dart dots
        """
        self.logger.info("Creating clean dartboard visualisation")
        
        if not score_result['success'] or score_result['dartboard_info'] is None:
            self.logger.warning("Cannot create visualisation without dartboard")
            # Return blank white image
            return np.ones_like(image) * 255
        
        segment_info = score_result['segment_info']
        centre = segment_info['centre']
        radius = segment_info['radius']
        
        # Create clean white background
        result = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        
        # Draw dartboard outline (black and white)
        self._draw_dartboard_outline(result, segment_info)
        
        # Draw dart positions as simple dots
        dart_scores = score_result['dart_scores']
        for dart_score in dart_scores:
            self._draw_dart_dot(result, dart_score)
        
        # Draw minimal summary
        self._draw_minimal_summary(result, score_result)
        
        if output_path:
            cv2.imwrite(output_path, result)
            self.logger.info(f"Saved visualisation to {output_path}")
        
        return result
    
    def _draw_dartboard_outline(self, image: np.ndarray, segment_info: Dict):
        """Draw a clean black and white dartboard outline."""
        centre = segment_info['centre']
        radius = segment_info['radius']
        
        black = (0, 0, 0)
        
        # Draw scoring rings (black outlines only)
        ring_info = [
            (DartboardDetector.RATIO_OUTER_DOUBLE, 2),   # Outer edge
            (DartboardDetector.RATIO_INNER_DOUBLE, 1),   # Double ring
            (DartboardDetector.RATIO_OUTER_TRIPLE, 1),   # Triple ring outer
            (DartboardDetector.RATIO_INNER_TRIPLE, 1),   # Triple ring inner
            (DartboardDetector.RATIO_OUTER_BULL, 1),     # Outer bull
            (DartboardDetector.RATIO_INNER_BULL, 1),     # Inner bull (bullseye)
        ]
        
        for ratio, thickness in ring_info:
            r = int(radius * ratio)
            cv2.circle(image, centre, r, black, thickness)
        
        # Draw segment boundaries (radial lines)
        segment_boundary_start = -90 - (DartboardDetector.SEGMENT_ANGLE / 2)
        
        for i in range(DartboardDetector.NUM_SEGMENTS):
            angle = segment_boundary_start + (i * DartboardDetector.SEGMENT_ANGLE)
            rad = np.radians(angle)
            x_end = int(centre[0] + radius * np.cos(rad))
            y_end = int(centre[1] + radius * np.sin(rad))
            cv2.line(image, centre, (x_end, y_end), black, 1)
        
        # Draw centre point
        cv2.circle(image, centre, 3, black, -1)
    
    def _draw_dart_dot(self, image: np.ndarray, dart_score: Dict):
        """Draw a simple dot to represent dart position."""
        position = dart_score['position']
        dart_id = dart_score['dart_id']
        
        black = (0, 0, 0)
        
        # Draw small black dot for dart position
        cv2.circle(image, position, 8, black, -1)
        
        # Optional: Add tiny number label
        font = cv2.FONT_HERSHEY_SIMPLEX
        dart_label = str(dart_id)
        (text_width, text_height), _ = cv2.getTextSize(dart_label, font, 0.4, 1)
        text_x = position[0] - text_width // 2
        text_y = position[1] + text_height // 2
        cv2.putText(image, dart_label, (text_x, text_y), font, 0.4, (255, 255, 255), 1)
    
    def _draw_minimal_summary(self, image: np.ndarray, score_result: Dict):
        """Draw a minimal score summary."""
        total_score = score_result['total_score']
        num_darts = len(score_result['dart_scores'])
        
        # Position at top of image
        summary_x = 10
        summary_y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        black = (0, 0, 0)
        
        # Simple text summary
        cv2.putText(image, f"Darts: {num_darts}  Total Score: {total_score}", 
                   (summary_x, summary_y), font, 0.7, black, 2)
        
        # List individual dart scores below
        y_offset = summary_y + 30
        for dart_score in score_result['dart_scores']:
            score_text = f"#{dart_score['dart_id']}: {dart_score['total']} pts ({dart_score['region']})"
            cv2.putText(image, score_text, (summary_x, y_offset), 
                       font, 0.5, black, 1)
            y_offset += 25
    
    def process_and_save(self, image_path: str, output_path: str = None) -> Dict:
        """
        Convenience method to process an image file and save the result.
        
        Args:
            image_path: Path to input image
            output_path: Optional path for output visualisation (defaults to input_scored.jpg)
            
        Returns:
            Score result dictionary
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return {
                'total_score': 0,
                'dart_scores': [],
                'darts': [],
                'dartboard_info': None,
                'success': False,
                'message': 'Could not load image'
            }
        
        # Score the image
        result = self.score_image(image)
        
        # Create visualisation
        if result['success']:
            if output_path is None:
                # Default output path
                base = image_path.rsplit('.', 1)[0]
                output_path = f"{base}_scored.jpg"
            
            self.visualise_score(image, result, output_path)
        
        return result

