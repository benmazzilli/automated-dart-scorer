# Automated Dart Scorer

A computer vision solution for automated dartboard detection, segment identification, and dart scoring using OpenCV.

## Project Overview

This project implements advanced computer vision techniques to:
- Detect dartboards in images using multiple detection methods
- Identify all 20 segments with accurate boundaries
- Detect scoring rings (doubles, triples, bullseye)
- Calculate scores for dart positions on the board
- Automatically detect board rotation and alignment

## Features

### Multi-Method Dartboard Detection
1. **Colour-based detection** - Fast detection using orange surround colour (Winmau boards)
2. **Contour-based detection** - Robust to lighting variations, finds circular shapes
3. **Hough circle detection** - Fallback method for difficult conditions

### Precise Segment Detection
- Detects all 20 dartboard segments with standard scoring layout
- Automatically aligns with board rotation using wire detection
- Identifies all scoring rings:
  - Inner bullseye (50 points)
  - Outer bull (25 points)
  - Triple ring (3x multiplier)
  - Double ring (2x multiplier)
  - Single regions (1x multiplier)

### Accurate Centre Detection
- Bullseye colour detection (red/green)
- Multiple refinement techniques for sub-pixel accuracy
- Validates centre position against detected rings

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

### Dependencies
- Python >= 3.13
- OpenCV (opencv-python >= 4.8.0)
- NumPy >= 1.24.0
- SciPy >= 1.16.2 (for advanced rotation detection)
- pytesseract >= 0.3.10

## Usage

### Running Tests

Test the dartboard detector with the provided test images:

```bash
python test_dartboard_detector.py
```

This will:
1. Load test images from `app/images/`
2. Detect the dartboard in each image
3. Identify all segments and scoring rings
4. Create visualizations with segment labels
5. Test scoring at sample dart positions
6. Save annotated images as `*_segments.jpg`

### Using the DartboardDetector Class

```python
from app.services.dartboard_detector import DartboardDetector
import cv2

# Initialize detector
detector = DartboardDetector(debug=True)

# Load image
image = cv2.imread('path/to/dartboard.jpg')

# Detect dartboard
dartboard_info = detector.detect_dartboard(image)

if dartboard_info:
    print(f"Dartboard found at {dartboard_info['centre']}")
    print(f"Radius: {dartboard_info['radius']}px")
    
    # Detect segments
    segment_info = detector.detect_segments(image, dartboard_info)
    segment_info.update(dartboard_info)
    
    # Create visualization
    result = detector.visualise_segments(image, segment_info, 'output.jpg')
    
    # Score a dart at specific coordinates
    dart_position = (500, 300)  # x, y coordinates
    score_info = detector.get_segment_score(dart_position, segment_info)
    print(f"Score: {score_info['total']} points")
    print(f"Region: {score_info['region']}")
    print(f"Base score: {score_info['score']} x {score_info['multiplier']}")
```

## How It Works

### 1. Dartboard Detection
The system tries multiple detection methods in order of speed and reliability:

1. **Colour Detection** (fastest)
   - Converts image to HSV colour space
   - Detects orange surround ring
   - Calculates centre using moments
   - Refines centre by detecting bullseye colours

2. **Contour Detection** (robust)
   - Applies edge detection with CLAHE enhancement
   - Finds circular contours
   - Scores by circularity and size

3. **Hough Circles** (fallback)
   - Uses Hough Circle Transform
   - Selects largest detected circle

### 2. Centre Refinement
- Searches for red/green bullseye colours in the centre region
- Uses Hough circles as fallback
- Achieves sub-pixel accuracy for precise segment boundaries

### 3. Rotation Detection
- Samples edge intensity along multiple radii
- Detects wire positions using peak detection
- Calculates rotation offset from expected positions
- Aligns segment boundaries to actual board orientation

### 4. Segment Mapping
- Divides board into 20 equal segments (18° each)
- Maps to standard dartboard score layout:
  ```
  20 at top, then clockwise: 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
  ```

### 5. Scoring Calculation
For any point (x, y):
1. Calculate distance from centre to determine ring
2. Calculate angle to determine segment
3. Apply multiplier based on ring:
   - Inner bull: 50 points
   - Outer bull: 25 points
   - Triple ring: base score × 3
   - Double ring: base score × 2
   - Single: base score × 1

## Project Structure

```
automated-dart-scorer/
├── app/
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── dartboard_detector.py    # Main detector implementation
│   └── images/                       # Test images
│       ├── test1.jpg
│       ├── test1_segments.jpg        # Generated output
│       ├── test2.jpg
│       └── test2_segments.jpg        # Generated output
├── test_dartboard_detector.py        # Test suite
├── pyproject.toml                    # Project dependencies
├── uv.lock                           # Locked dependencies
└── README.md                         # This file
```

## Configuration

The `DartboardDetector` class uses standard dartboard dimensions based on official specifications:

- **Playing diameter**: 451mm (outer edge of double ring)
- **Double ring**: 162mm from centre (inner edge)
- **Triple ring**: 99-107mm from centre
- **Outer bull**: 15.9mm radius
- **Inner bull**: 6.35mm radius

These are automatically scaled to the detected board size in the image.

## Example Output

When running tests, you'll see output like:

```
🎯 Dartboard Detector Test Suite
======================================================================
Testing: Test 1 - First Dartboard
======================================================================
📷 Loading image: app/images/test1.jpg
✓ Image loaded: 1920x1080 pixels

🔍 Detecting dartboard...
✓ Dartboard detected!
  Centre: (960, 540)
  Radius: 450px
  Method: colour

🎯 Detecting segments...
✓ Segments detected!
  Number of segments: 20

🎨 Creating visualisation...
Detected 20 wires, rotation offset: 2.3°
✓ Visualisation saved to: app/images/test1_segments.jpg

🎯 Testing dart scoring:
  Test point 1 at (960, 315):
    Score: 20
    Region: triple
    Multiplier: 3x
    Total: 60 points
```

## Current Status

This project is under active development on the `feature/computer-vision-scoring` branch.

### Completed
- ✅ Multi-method dartboard detection
- ✅ Precise centre detection with bullseye refinement
- ✅ Automatic rotation detection and alignment
- ✅ Segment boundary detection
- ✅ Scoring ring identification
- ✅ Point-to-score calculation
- ✅ Comprehensive test suite
- ✅ Visual debugging and annotation

### Future Enhancements
- 🔄 Dart tip detection from images
- 🔄 Video stream processing
- 🔄 Real-time scoring interface
- 🔄 Game mode implementation (501, Cricket, etc.)
- 🔄 Multi-player support
- 🔄 Statistics tracking

## Troubleshooting

**No dartboard detected:**
- Ensure the dartboard is clearly visible and well-lit
- Try different detection methods by checking debug output
- The dartboard should be roughly circular in the image

**Incorrect segment boundaries:**
- Check that the centre detection is accurate (visualize output)
- Ensure wire detection is working (check debug logs)
- May need to adjust rotation detection parameters for non-standard boards

**Wrong scores:**
- Verify centre position is accurate
- Check that the radius detection is correct
- Ensure the board is not significantly distorted in the image

## Contributing

This is an active development project. Contributions and suggestions are welcome!

## License

This project is for educational and personal use.
